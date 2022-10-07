/*
	Copyright (C) Georgy Firsov. 2022.

	XEH mode of operation implementation.
	Implemented using Intel intrinsics.

	Signatures copied from XTS mode implementation
	due to public interface compatibility between
	XEH and XTS.
*/

/*

TODO:
- Implement XEH encryption instead of XTS algorithm
- Implement XEH decryption instead of XTS algorithm
- Fix decryption key generation

*/

#ifdef TC_MINIMIZE_CODE_SIZE
//	Preboot/boot version
#	ifndef TC_NO_COMPILER_INT64
#		define TC_NO_COMPILER_INT64
#	endif
#	pragma optimize ("tl", on)
#endif

#ifdef TC_NO_COMPILER_INT64
#	include <memory.h>
#endif

#ifndef TC_NO_COMPILER_INT64
#include "cpu.h"
#include "misc.h"
#endif
#include "Xeh.h"


#include <wmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>


#define ALIGN16 CRYPTOPP_ALIGN_DATA(16)

//
// IEEE P1619/D16 Annex C
// x ^ 128 + x ^ 7 + x ^ 2 + x + 1
//           ---------------------
//
#define GF_128_FDBK 0x87

//
// Size of GF(2 ^ 128) value
//
#define GF_128_SIZE 16


__forceinline __m128i gf128_multiply(__m128i lhs, __m128i rhs)
{
    //
    // Intel Carry-Less Multiplication Instruction and its Usage for Computing the GCM Mode
    //
    // Algorithm 2 (Carry-less multiplication of 128-bit operands using PCLMULQDQ) 
    // Algorithm 4 (Application of the method for reduction modulo x^128 + x^7 + x^2 + x + 1)
    //

    __m128i tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10;
    __m128i mask = _mm_setr_epi32(0xffffffff, 0, 0, 0);

    tmp1  = _mm_clmulepi64_si128(lhs, rhs, 0x00);
    tmp4  = _mm_clmulepi64_si128(lhs, rhs, 0x11);

    tmp2  = _mm_shuffle_epi32(lhs,78); 
    tmp3  = _mm_shuffle_epi32(rhs,78); 
    tmp2  = _mm_xor_si128(tmp2, lhs); 
    tmp3  = _mm_xor_si128(tmp3, rhs);

    tmp2  = _mm_clmulepi64_si128(tmp2, tmp3, 0x00);

    tmp2  = _mm_xor_si128(tmp2, tmp1); 
    tmp2  = _mm_xor_si128(tmp2, tmp4);

    tmp3  = _mm_slli_si128(tmp2, 8); 
    tmp2  = _mm_srli_si128(tmp2, 8); 
    tmp1  = _mm_xor_si128(tmp1, tmp3); 
    tmp4  = _mm_xor_si128(tmp4, tmp2);

    tmp5  = _mm_srli_epi32(tmp4, 31); 
    tmp6  = _mm_srli_epi32(tmp4, 30); 
    tmp7  = _mm_srli_epi32(tmp4, 25);

    tmp5  = _mm_xor_si128(tmp5, tmp6); 
    tmp5  = _mm_xor_si128(tmp5, tmp7);

    tmp6  = _mm_shuffle_epi32(tmp5, 147);

    tmp5  = _mm_and_si128(mask, tmp6); 
    tmp6  = _mm_andnot_si128(mask, tmp6); 
    tmp1  = _mm_xor_si128(tmp1, tmp6);
    tmp4  = _mm_xor_si128(tmp4, tmp5);

    tmp8  = _mm_slli_epi32(tmp4, 1);
    tmp1  = _mm_xor_si128(tmp1, tmp8); 
    tmp9  = _mm_slli_epi32(tmp4, 2); 
    tmp1  = _mm_xor_si128(tmp1, tmp9); 
    tmp10 = _mm_slli_epi32(tmp4, 7);
    tmp1  = _mm_xor_si128(tmp1, tmp10);

    return _mm_xor_si128(tmp1, tmp4);
}


__forceinline __m128i gf128_multiply_primitive(__m128i op)
{
    unsigned char           carry_in;
    unsigned char           carry_out;
    unsigned int            idx;
    ALIGN16 unsigned char   internal_op[GF_128_SIZE];

    _mm_store_si128((__m128i *)internal_op, op);

    carry_in = 0;
    for (idx = 0; idx < GF_128_SIZE; ++idx)
    {
        carry_out = (internal_op[idx] >> 7) & 0xff;
        internal_op[idx] = ((internal_op[idx] << 1) + carry_in) & 0xff;
        carry_in = carry_out;
    }

    if (carry_out)
    {
        internal_op[0] ^= GF_128_FDBK;
    }

    return _mm_load_si128((const __m128i *)internal_op);
}


__forceinline void xehaesp_tweaks_init(unsigned long long sector, 
									   unsigned __int8 *first_key, 
									   unsigned __int8 *second_key,
									   __m128i *tweak1, 
									   __m128i *tweak2, 
									   __m128i *tweak3,
									   int cipher)
{
    __m128i                 internal_sector;
    unsigned int            idx;
    ALIGN16 unsigned char   internal_tweak[BYTES_PER_XEH_BLOCK];

    for (idx = 0; idx < sizeof(internal_tweak); ++idx)
    {
        internal_tweak[idx] = (unsigned char)(sector & 0xFF);
        sector >>= 8;
    }

    internal_sector = _mm_load_si128((const __m128i *)internal_tweak);
    
	EncipherBlockEx (cipher, &internal_sector, tweak1, first_key);
	EncipherBlockEx (cipher, tweak1,           tweak2, second_key);
	EncipherBlockEx (cipher, &internal_sector, tweak3, second_key);
}


__forceinline void xehaesp_apply_f_inverse(const unsigned char *in, 
										   unsigned long blocks, 
										   __m128i tweak3, 
										   unsigned char *out)
{
    //
    // Implementation of inverse of f
    //

    __m128i         Y;
    __m128i         accumulated_tweak;
    __m128i         immediate;
    __m128i         temp;
    unsigned int    block;

    const __m128i   *internal_in  = (const __m128i *)in;
    __m128i         *internal_out = (__m128i *)out;

    //
    // First recover n - 1 blocks
    //

    immediate = _mm_loadu_si128(&internal_in[blocks - 1]);

    for (block = 0; block < blocks - 1; ++block)
    {
        internal_out[block] = _mm_xor_si128(internal_in[block], immediate);
    }

    //
    // Now recover the last one
    //

    Y                 = _mm_setr_epi32(0, 0, 0, 0);
    accumulated_tweak = tweak3;

    for (block = blocks - 1; block > 0; --block)
    {
        //
        // Calculate current multiplier and multtiply by current block
        //

        temp              = _mm_setr_epi32(block, 0, 0, 0);
        temp              = _mm_xor_si128(temp, accumulated_tweak);
        temp              = gf128_multiply(internal_out[block - 1], temp);

        //
        // Add to immediate value and get next power of tweak
        //

        Y                 = _mm_xor_si128(Y, temp);
        accumulated_tweak = gf128_multiply(accumulated_tweak, tweak3);
    }

    internal_out[blocks - 1] = _mm_xor_si128(internal_in[blocks - 1], Y);
}


__forceinline void xehaesp_apply_f(const unsigned char *in, 
								   unsigned long blocks, 
								   __m128i tweak3, 
								   unsigned char *out)
{
    //
    // Implementation of f permutation:
    // f(x[1], ..., x[n]) = (x[1] + Y, ..., x[n-1] + Y, Y)
    //
    // Y = x[n] + x[n - 1] * (tweak + n - 1) + ... + x[1] * (tweak ^ {n - 1} + 1)
    //

    __m128i         Y;
    __m128i         accumulated_tweak;
    __m128i         temp1;
    __m128i         temp2;
    unsigned int    block;

    const __m128i   *internal_in  = (const __m128i *)in;
    __m128i         *internal_out = (__m128i *)out;

    //
    // Calculate Y value of f
    //

    Y                 = _mm_setr_epi32(0, 0, 0, 0);
    accumulated_tweak = tweak3;

    for (block = blocks - 1; block > 0; --block)
    {
        //
        // Load (n - j)-th block into register
        // We need to subtract 1 from 'block' here, because
        // indices are shifted
        //

        temp1             = _mm_loadu_si128(&internal_in[block - 1]);

        //
        // Calculate current multiplier and multtiply by current block
        //

        temp2             = _mm_setr_epi32(block, 0, 0, 0);
        temp2             = _mm_xor_si128(temp2, accumulated_tweak);
        temp1             = gf128_multiply(temp1, temp2);

        //
        // Add to Y and get next power of tweak
        //

        Y                 = _mm_xor_si128(Y, temp1);
        accumulated_tweak = gf128_multiply(accumulated_tweak, tweak3);
    }

    //
    // Finally add the last block
    //

    temp1 = _mm_loadu_si128(&internal_in[blocks - 1]);
    Y     = _mm_xor_si128(Y, temp1);

    //
    // Apply f transformation
    //

    for (block = 0; block < blocks - 1; ++block)
    {
        internal_out[block] = _mm_xor_si128(internal_in[block], Y);
    }

    internal_out[blocks - 1] = Y;
}



// length: number of bytes to encrypt; may be larger than one data unit and must be divisible by the cipher block size
// ks: the primary key schedule
// ks2: the secondary key schedule
// startDataUnitNo: The sequential number of the data unit with which the buffer starts.
// startCipherBlockNo: The sequential number of the first plaintext block to encrypt inside the data unit startDataUnitNo.
//                     When encrypting the data unit from its first block, startCipherBlockNo is 0.
//                     The startCipherBlockNo value applies only to the first data unit in the buffer; each successive
//                     data unit is encrypted from its first block. The start of the buffer does not have to be
//                     aligned with the start of a data unit. If it is aligned, startCipherBlockNo must be 0; if it
//                     is not aligned, startCipherBlockNo must reflect the misalignment accordingly.
void EncryptBufferXEH (unsigned __int8 *buffer,
					   TC_LARGEST_COMPILER_UINT length,
					   const UINT64_STRUCT *startDataUnitNo,
					   unsigned int startCipherBlockNo,
					   unsigned __int8 *ks,
					   unsigned __int8 *ks2,
					   int cipher)
{
	unsigned __int8 finalCarry;
	CRYPTOPP_ALIGN_DATA(16) unsigned __int8 whiteningValue [BYTES_PER_XEH_BLOCK];
	unsigned __int8 byteBufUnitNo [BYTES_PER_XEH_BLOCK];
	unsigned __int64 *whiteningValuePtr64 = (unsigned __int64 *) whiteningValue;
	unsigned __int64 *bufPtr = (unsigned __int64 *) buffer;
	unsigned int startBlock = startCipherBlockNo, endBlock, block;
	TC_LARGEST_COMPILER_UINT blockCount, dataUnitNo;

	/* The encrypted data unit number (i.e. the resultant ciphertext block) is to be multiplied in the
	finite field GF(2^128) by j-th power of n, where j is the sequential plaintext/ciphertext block
	number and n is 2, a primitive element of GF(2^128). This can be (and is) simplified and implemented
	as a left shift of the preceding whitening value by one bit (with carry propagating). In addition, if
	the shift of the highest byte results in a carry, 135 is XORed into the lowest byte. The value 135 is
	derived from the modulus of the Galois Field (x^128+x^7+x^2+x+1). */

	// Convert the 64-bit data unit number into a little-endian 16-byte array.
	// Note that as we are converting a 64-bit number into a 16-byte array we can always zero the last 8 bytes.
	dataUnitNo = startDataUnitNo->Value;
	*((unsigned __int64 *) byteBufUnitNo) = LE64 (dataUnitNo);
	*((unsigned __int64 *) byteBufUnitNo + 1) = 0;

	if (length % BYTES_PER_XEH_BLOCK)
		TC_THROW_FATAL_EXCEPTION;

	blockCount = length / BYTES_PER_XEH_BLOCK;

	// Process all blocks in the buffer
	while (blockCount > 0)
	{
		//
		// Encrypt several sectors (data units) here
		//

		if (blockCount < BLOCKS_PER_XEH_DATA_UNIT)
			endBlock = startBlock + (unsigned int) blockCount;
		else
			endBlock = BLOCKS_PER_XEH_DATA_UNIT;

		whiteningValuePtr64 = (unsigned __int64 *) whiteningValue;

		// Encrypt the data unit number using the secondary key (in order to generate the first
		// whitening value for this data unit)
		*whiteningValuePtr64 = *((unsigned __int64 *) byteBufUnitNo);
		*(whiteningValuePtr64 + 1) = 0;
		EncipherBlock (cipher, whiteningValue, ks2);

		// Generate (and apply) subsequent whitening values for blocks in this data unit and
		// encrypt all relevant blocks in this data unit
		for (block = 0; block < endBlock; block++)
		{
			if (block >= startBlock)
			{
				// Pre-whitening
				__m128i xmm1 = _mm_loadu_si128((const __m128i*)whiteningValuePtr64);
				__m128i xmm2 = _mm_loadu_si128((__m128i*)bufPtr);

				_mm_storeu_si128((__m128i*)bufPtr, _mm_xor_si128(xmm1, xmm2));

				// Actual encryption
				EncipherBlock (cipher, bufPtr, ks);

				// Post-whitening
				xmm1 = _mm_loadu_si128((const __m128i*)whiteningValuePtr64);
				xmm2 = _mm_loadu_si128((__m128i*)bufPtr);

				_mm_storeu_si128((__m128i*)bufPtr, _mm_xor_si128(xmm1, xmm2));
				
				whiteningValuePtr64++;
				bufPtr += 2;
			}
			else
				whiteningValuePtr64++;

			//
			// Derive the next whitening value
			//

			finalCarry =
				(*whiteningValuePtr64 & 0x8000000000000000) ?
				135 : 0;

			*whiteningValuePtr64-- <<= 1;

			if (*whiteningValuePtr64 & 0x8000000000000000)
				*(whiteningValuePtr64 + 1) |= 1;

			*whiteningValuePtr64 <<= 1;


			whiteningValue[0] ^= finalCarry;
		}

		blockCount -= endBlock - startBlock;
		startBlock = 0;
		dataUnitNo++;
		*((unsigned __int64 *) byteBufUnitNo) = LE64 (dataUnitNo);
	}

	FAST_ERASE64 (whiteningValue, sizeof (whiteningValue));
}


void DecryptBufferXEH (unsigned __int8 *buffer,
					   TC_LARGEST_COMPILER_UINT length,
					   const UINT64_STRUCT *startDataUnitNo,
					   unsigned int startCipherBlockNo,
					   unsigned __int8 *ks,
					   unsigned __int8 *ks2,
					   int cipher)
{
	unsigned __int8 finalCarry;
	unsigned __int8 whiteningValue [BYTES_PER_XEH_BLOCK];
	unsigned __int8 byteBufUnitNo [BYTES_PER_XEH_BLOCK];
	unsigned __int64 *whiteningValuePtr64 = (unsigned __int64 *) whiteningValue;
	unsigned __int64 *bufPtr = (unsigned __int64 *) buffer;
	unsigned int startBlock = startCipherBlockNo, endBlock, block;
	TC_LARGEST_COMPILER_UINT blockCount, dataUnitBlockCount, dataUnitNo;

	// Convert the 64-bit data unit number into a little-endian 16-byte array.
	// Note that as we are converting a 64-bit number into a 16-byte array we can always zero the last 8 bytes.
	dataUnitNo = startDataUnitNo->Value;
	*((unsigned __int64 *) byteBufUnitNo) = LE64 (dataUnitNo);
	*((unsigned __int64 *) byteBufUnitNo + 1) = 0;

	if (length % BYTES_PER_XEH_BLOCK)
		TC_THROW_FATAL_EXCEPTION;

	blockCount = length / BYTES_PER_XEH_BLOCK;

	// Process all blocks in the buffer
	while (blockCount > 0)
	{
		//
		// Decrypt several sectors (data units) here
		//

		if (blockCount < BLOCKS_PER_XEH_DATA_UNIT)
			endBlock = startBlock + (unsigned int) blockCount;
		else
			endBlock = BLOCKS_PER_XEH_DATA_UNIT;

		//
		// Derive tweak values
		// BUGBUG: here we need BOTH ks and ks2 to be ENCRYPTION keys, not decryption ones.
		// Need to fix key initialization
		//
		// Example:
		//    aes128_initalize_decrypt_key(data_key,  &internal_data_key);   // ks  for data decryption
		//    aes128_initalize_encrypt_key(data_key,  &internal_tweak_key1); // ks  for tweak derivation
		//    aes128_initalize_encrypt_key(tweak_key, &internal_tweak_key2); // ks2 for tweak derivation
		//

		// xehaesp_tweaks_init(dataUnitNo, ks, ks2, &tweak1, &tweak2, &tweak3, cipher);

		//
		// Apply f transformation
		//

		dataUnitBlockCount = endBlock - startBlock;
		// xehaesp_apply_f(bufPtr, (unsigned long)dataUnitBlockCount, tweak3, bufPtr);

		//
		// Now comes decryption itself
		//

		// Generate (and apply) subsequent whitening values for blocks in this data unit and
		// decrypt all relevant blocks in this data unit
		for (block = 0; block < endBlock; block++)
		{
			if (block >= startBlock)
			{
				// Post-whitening
				__m128i xmm1 = _mm_loadu_si128((const __m128i*)whiteningValuePtr64);
				__m128i xmm2 = _mm_loadu_si128((__m128i*)bufPtr);

				_mm_storeu_si128((__m128i*)bufPtr, _mm_xor_si128(xmm1, xmm2));

				// Actual decryption
				DecipherBlock (cipher, bufPtr, ks);

				// Pre-whitening
				xmm1 = _mm_loadu_si128((const __m128i*)whiteningValuePtr64);
				xmm2 = _mm_loadu_si128((__m128i*)bufPtr);

				_mm_storeu_si128((__m128i*)bufPtr, _mm_xor_si128(xmm1, xmm2));
				
				whiteningValuePtr64++;
				bufPtr += 2;
			}
			else
				whiteningValuePtr64++;

			// Derive the next whitening value

			finalCarry =
				(*whiteningValuePtr64 & 0x8000000000000000) ?
				135 : 0;

			*whiteningValuePtr64-- <<= 1;

			if (*whiteningValuePtr64 & 0x8000000000000000)
				*(whiteningValuePtr64 + 1) |= 1;

			*whiteningValuePtr64 <<= 1;

			whiteningValue[0] ^= finalCarry;
		}

		blockCount -= dataUnitBlockCount;
		startBlock = 0;
		dataUnitNo++;
		*((unsigned __int64 *) byteBufUnitNo) = LE64 (dataUnitNo);
	}

	FAST_ERASE64 (whiteningValue, sizeof (whiteningValue));
}
