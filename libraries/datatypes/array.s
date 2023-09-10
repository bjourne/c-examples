	.file	"array.c"
	.intel_syntax noprefix
	.text
	.p2align 4
	.type	cmp_fun, @function
cmp_fun:
.LFB12:
	.cfi_startproc
	mov	rcx, QWORD PTR [rdx]
	mov	rax, QWORD PTR 8[rdx]
	mov	rsi, QWORD PTR [rsi]
	mov	rdi, QWORD PTR [rdi]
	mov	rdx, rcx
	jmp	rax
	.cfi_endproc
.LFE12:
	.size	cmp_fun, .-cmp_fun
	.p2align 4
	.type	key_cmp_fun, @function
key_cmp_fun:
.LFB14:
	.cfi_startproc
	push	r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	mov	r12, rsi
	push	rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	push	rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
	mov	rbx, rdx
	mov	rsi, QWORD PTR [rdx]
	call	[QWORD PTR 8[rdx]]
	mov	rsi, QWORD PTR [rbx]
	mov	rdi, r12
	mov	ebp, eax
	call	[QWORD PTR 8[rbx]]
	pop	rbx
	.cfi_def_cfa_offset 24
	mov	edx, eax
	mov	eax, ebp
	pop	rbp
	.cfi_def_cfa_offset 16
	pop	r12
	.cfi_def_cfa_offset 8
	sub	eax, edx
	ret
	.cfi_endproc
.LFE14:
	.size	key_cmp_fun, .-key_cmp_fun
	.p2align 4
	.type	ind_cmp_fun, @function
ind_cmp_fun:
.LFB16:
	.cfi_startproc
	mov	rcx, QWORD PTR 24[rdx]
	mov	esi, esi
	mov	edi, edi
	mov	rax, QWORD PTR 16[rdx]
	mov	r8, QWORD PTR [rdx]
	imul	rsi, rcx
	imul	rdi, rcx
	mov	rsi, QWORD PTR [rax+rsi]
	mov	rdi, QWORD PTR [rax+rdi]
	mov	rax, QWORD PTR 8[rdx]
	mov	rdx, r8
	jmp	rax
	.cfi_endproc
.LFE16:
	.size	ind_cmp_fun, .-ind_cmp_fun
	.p2align 4
	.globl	array_ord_asc_u32
	.type	array_ord_asc_u32, @function
array_ord_asc_u32:
.LFB20:
	.cfi_startproc
	mov	eax, 1
	cmp	esi, edi
	jb	.L6
	cmp	edi, esi
	sbb	eax, eax
.L6:
	ret
	.cfi_endproc
.LFE20:
	.size	array_ord_asc_u32, .-array_ord_asc_u32
	.p2align 4
	.globl	array_ord_asc_i32
	.type	array_ord_asc_i32, @function
array_ord_asc_i32:
.LFB21:
	.cfi_startproc
	mov	eax, edi
	sub	eax, esi
	ret
	.cfi_endproc
.LFE21:
	.size	array_ord_asc_i32, .-array_ord_asc_i32
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC0:
	.string	"array.c"
.LC1:
	.string	"size <= 128"
	.text
	.p2align 4
	.globl	array_shuffle
	.type	array_shuffle, @function
array_shuffle:
.LFB11:
	.cfi_startproc
	push	r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	push	r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	push	r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	push	r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	push	rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	push	rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	sub	rsp, 168
	.cfi_def_cfa_offset 224
	mov	rax, QWORD PTR fs:40
	mov	QWORD PTR 152[rsp], rax
	xor	eax, eax
	cmp	rdx, 128
	ja	.L18
	mov	r14, rsi
	cmp	rsi, 1
	jbe	.L10
	lea	rax, -1[rsi]
	mov	r15, rdi
	mov	rbp, rdx
	mov	r12, rdi
	mov	QWORD PTR 8[rsp], rax
	lea	rax, 16[rsp]
	xor	r13d, r13d
	mov	QWORD PTR [rsp], rax
	.p2align 4,,10
	.p2align 3
.L13:
	call	rand@PLT
	mov	rsi, r14
	xor	edx, edx
	mov	rdi, QWORD PTR [rsp]
	movsx	rbx, eax
	sub	rsi, r13
	mov	eax, 2147483647
	div	rsi
	xor	edx, edx
	lea	rsi, 1[rax]
	mov	rax, rbx
	div	rsi
	mov	rdx, rbp
	mov	rbx, rax
	add	rbx, r13
	inc	r13
	imul	rbx, rbp
	add	rbx, r15
	mov	rsi, rbx
	call	memcpy@PLT
	mov	rsi, r12
	mov	rdx, rbp
	mov	rdi, rbx
	call	memcpy@PLT
	mov	rsi, QWORD PTR [rsp]
	mov	rdi, r12
	mov	rdx, rbp
	add	r12, rbp
	call	memcpy@PLT
	cmp	QWORD PTR 8[rsp], r13
	jne	.L13
.L10:
	mov	rax, QWORD PTR 152[rsp]
	sub	rax, QWORD PTR fs:40
	jne	.L19
	add	rsp, 168
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	pop	rbx
	.cfi_def_cfa_offset 48
	pop	rbp
	.cfi_def_cfa_offset 40
	pop	r12
	.cfi_def_cfa_offset 32
	pop	r13
	.cfi_def_cfa_offset 24
	pop	r14
	.cfi_def_cfa_offset 16
	pop	r15
	.cfi_def_cfa_offset 8
	ret
.L18:
	.cfi_restore_state
	lea	rcx, __PRETTY_FUNCTION__.1[rip]
	mov	edx, 14
	lea	rsi, .LC0[rip]
	lea	rdi, .LC1[rip]
	call	__assert_fail@PLT
.L19:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE11:
	.size	array_shuffle, .-array_shuffle
	.p2align 4
	.globl	array_qsort
	.type	array_qsort, @function
array_qsort:
.LFB13:
	.cfi_startproc
	sub	rsp, 40
	.cfi_def_cfa_offset 48
	mov	rax, QWORD PTR fs:40
	mov	QWORD PTR 24[rsp], rax
	xor	eax, eax
	mov	QWORD PTR [rsp], r8
	mov	r8, rsp
	mov	QWORD PTR 8[rsp], rcx
	lea	rcx, cmp_fun[rip]
	call	qsort_r@PLT
	mov	rax, QWORD PTR 24[rsp]
	sub	rax, QWORD PTR fs:40
	jne	.L24
	add	rsp, 40
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L24:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE13:
	.size	array_qsort, .-array_qsort
	.p2align 4
	.globl	array_qsort_by_key
	.type	array_qsort_by_key, @function
array_qsort_by_key:
.LFB15:
	.cfi_startproc
	sub	rsp, 56
	.cfi_def_cfa_offset 64
	mov	rax, QWORD PTR fs:40
	mov	QWORD PTR 40[rsp], rax
	xor	eax, eax
	mov	rax, rsp
	mov	QWORD PTR [rsp], r8
	lea	r8, 16[rsp]
	mov	QWORD PTR 16[rsp], rax
	lea	rax, key_cmp_fun[rip]
	mov	QWORD PTR 8[rsp], rcx
	lea	rcx, cmp_fun[rip]
	mov	QWORD PTR 24[rsp], rax
	xor	eax, eax
	call	qsort_r@PLT
	mov	rax, QWORD PTR 40[rsp]
	sub	rax, QWORD PTR fs:40
	jne	.L29
	add	rsp, 56
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L29:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE15:
	.size	array_qsort_by_key, .-array_qsort_by_key
	.p2align 4
	.globl	array_qsort_indirect
	.type	array_qsort_indirect, @function
array_qsort_indirect:
.LFB17:
	.cfi_startproc
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	push	r15
	.cfi_offset 15, -24
	mov	r15, rsi
	push	r14
	.cfi_offset 14, -32
	mov	r14, r8
	push	r13
	.cfi_offset 13, -40
	mov	r13, rcx
	push	r12
	.cfi_offset 12, -48
	mov	r12, rdi
	lea	rdi, 0[0+rsi*8]
	push	rbx
	and	rsp, -32
	sub	rsp, 96
	.cfi_offset 3, -56
	mov	QWORD PTR 24[rsp], rdx
	mov	rax, QWORD PTR fs:40
	mov	QWORD PTR 88[rsp], rax
	xor	eax, eax
	call	malloc@PLT
	mov	rbx, rax
	test	r15, r15
	je	.L31
	lea	rax, -1[r15]
	cmp	rax, 2
	jbe	.L36
	mov	rdx, r15
	vmovdqa	ymm0, YMMWORD PTR .LC2[rip]
	mov	rax, rbx
	vpbroadcastq	ymm2, QWORD PTR .LC4[rip]
	shr	rdx, 2
	sal	rdx, 5
	add	rdx, rbx
	.p2align 4,,10
	.p2align 3
.L33:
	vmovdqa	ymm1, ymm0
	add	rax, 32
	vpaddq	ymm0, ymm0, ymm2
	vmovdqu	YMMWORD PTR -32[rax], ymm1
	cmp	rax, rdx
	jne	.L33
	test	r15b, 3
	je	.L44
	mov	rax, r15
	and	rax, -4
	vzeroupper
.L32:
	lea	rdx, 1[rax]
	mov	QWORD PTR [rbx+rax*8], rax
	lea	rcx, 0[0+rax*8]
	cmp	rdx, r15
	jnb	.L31
	add	rax, 2
	mov	QWORD PTR 8[rbx+rcx], rdx
	cmp	rax, r15
	jnb	.L31
	mov	QWORD PTR 16[rbx+rcx], rax
.L31:
	mov	rax, QWORD PTR 24[rsp]
	lea	r8, 32[rsp]
	mov	rsi, r15
	mov	rdi, rbx
	lea	rcx, cmp_fun[rip]
	mov	edx, 8
	mov	QWORD PTR 48[rsp], r14
	mov	QWORD PTR 72[rsp], rax
	lea	rax, 48[rsp]
	mov	QWORD PTR 32[rsp], rax
	lea	rax, ind_cmp_fun[rip]
	mov	QWORD PTR 40[rsp], rax
	xor	eax, eax
	mov	QWORD PTR 56[rsp], r13
	mov	QWORD PTR 64[rsp], r12
	call	qsort_r@PLT
	mov	rax, QWORD PTR 88[rsp]
	sub	rax, QWORD PTR fs:40
	jne	.L46
	lea	rsp, -40[rbp]
	mov	rax, rbx
	pop	rbx
	pop	r12
	pop	r13
	pop	r14
	pop	r15
	pop	rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L44:
	.cfi_restore_state
	vzeroupper
	jmp	.L31
.L36:
	xor	eax, eax
	jmp	.L32
.L46:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE17:
	.size	array_qsort_indirect, .-array_qsort_indirect
	.section	.rodata.str1.1
.LC5:
	.string	"false"
	.text
	.p2align 4
	.globl	array_permute
	.type	array_permute, @function
array_permute:
.LFB18:
	.cfi_startproc
	cmp	rsi, 1
	je	.L61
	mov	r8, rdx
	lea	r9, -1[rsi]
	xor	edx, edx
	.p2align 4,,10
	.p2align 3
.L55:
	mov	rax, QWORD PTR [rcx+rdx*8]
	cmp	rax, rdx
	jnb	.L49
	.p2align 4,,10
	.p2align 3
.L50:
	mov	rax, QWORD PTR [rcx+rax*8]
	cmp	rax, rdx
	jb	.L50
.L49:
	cmp	r8, 1
	je	.L64
	cmp	r8, 4
	je	.L65
	cmp	r8, 8
	jne	.L54
	lea	rax, [rdi+rax*8]
	mov	rsi, QWORD PTR [rdi+rdx*8]
	mov	r10, QWORD PTR [rax]
	mov	QWORD PTR [rdi+rdx*8], r10
	mov	QWORD PTR [rax], rsi
.L52:
	inc	rdx
	cmp	r9, rdx
	jne	.L55
.L61:
	ret
	.p2align 4,,10
	.p2align 3
.L64:
	add	rax, rdi
	movzx	esi, BYTE PTR [rdi+rdx]
	movzx	r10d, BYTE PTR [rax]
	mov	BYTE PTR [rdi+rdx], r10b
	mov	BYTE PTR [rax], sil
	jmp	.L52
	.p2align 4,,10
	.p2align 3
.L65:
	lea	rax, [rdi+rax*4]
	mov	esi, DWORD PTR [rdi+rdx*4]
	mov	r10d, DWORD PTR [rax]
	mov	DWORD PTR [rdi+rdx*4], r10d
	mov	DWORD PTR [rax], esi
	jmp	.L52
.L54:
	sub	rsp, 8
	.cfi_def_cfa_offset 16
	lea	rcx, __PRETTY_FUNCTION__.0[rip]
	mov	edx, 132
	lea	rsi, .LC0[rip]
	lea	rdi, .LC5[rip]
	call	__assert_fail@PLT
	.cfi_endproc
.LFE18:
	.size	array_permute, .-array_permute
	.p2align 4
	.globl	array_ord_asc_u8
	.type	array_ord_asc_u8, @function
array_ord_asc_u8:
.LFB19:
	.cfi_startproc
	movzx	eax, dil
	movzx	esi, sil
	sub	eax, esi
	ret
	.cfi_endproc
.LFE19:
	.size	array_ord_asc_u8, .-array_ord_asc_u8
	.p2align 4
	.globl	array_bsearch
	.type	array_bsearch, @function
array_bsearch:
.LFB24:
	.cfi_startproc
	push	r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	push	r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	mov	r14, rdi
	push	r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	mov	r13, rcx
	push	r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	mov	r12, rsi
	push	rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	push	rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	sub	rsp, 24
	.cfi_def_cfa_offset 80
	mov	QWORD PTR 8[rsp], r8
	mov	QWORD PTR [rsp], r9
	cmp	rdx, 4
	je	.L68
	mov	r15, rdx
	xor	ebx, ebx
	test	rsi, rsi
	je	.L67
	.p2align 4,,10
	.p2align 3
.L69:
	mov	rbp, r12
	shr	r12
	mov	rdx, QWORD PTR 8[rsp]
	mov	rsi, QWORD PTR [rsp]
	lea	rax, [r12+rbx]
	imul	rax, r15
	mov	rdi, QWORD PTR [r14+rax]
	call	r13
	test	eax, eax
	jns	.L81
	and	ebp, 1
	add	rbp, r12
	add	rbx, rbp
.L81:
	test	r12, r12
	jne	.L69
.L67:
	add	rsp, 24
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	mov	rax, rbx
	pop	rbx
	.cfi_def_cfa_offset 48
	pop	rbp
	.cfi_def_cfa_offset 40
	pop	r12
	.cfi_def_cfa_offset 32
	pop	r13
	.cfi_def_cfa_offset 24
	pop	r14
	.cfi_def_cfa_offset 16
	pop	r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L68:
	.cfi_restore_state
	lea	rax, array_ord_asc_i32[rip]
	cmp	rcx, rax
	je	.L94
	lea	rax, array_ord_asc_u32[rip]
	cmp	rcx, rax
	je	.L75
	xor	ebx, ebx
	test	rsi, rsi
	je	.L67
	.p2align 4,,10
	.p2align 3
.L76:
	mov	rbp, r12
	shr	r12
	mov	rdx, QWORD PTR 8[rsp]
	mov	rsi, QWORD PTR [rsp]
	lea	rax, [r12+rbx]
	mov	rdi, QWORD PTR [r14+rax*4]
	call	r13
	test	eax, eax
	jns	.L80
	and	ebp, 1
	add	rbp, r12
	add	rbx, rbp
.L80:
	test	r12, r12
	jne	.L76
	jmp	.L67
	.p2align 4,,10
	.p2align 3
.L75:
	mov	ecx, DWORD PTR [rsp]
	test	rsi, rsi
	je	.L85
	mov	rbx, rdi
	.p2align 4,,10
	.p2align 3
.L79:
	mov	rax, r12
	shr	r12
	mov	rdx, rax
	and	rax, -4
	sub	rdx, r12
	cmp	DWORD PTR [rbx+r12*4], ecx
	lea	rdx, [rbx+rdx*4]
	cmovb	rbx, rdx
	prefetcht0	[rdx+rax]
	test	r12, r12
	jne	.L79
.L93:
	sub	rbx, r14
	sar	rbx, 2
	jmp	.L67
	.p2align 4,,10
	.p2align 3
.L94:
	mov	edx, DWORD PTR [rsp]
	test	rsi, rsi
	je	.L85
	mov	rbx, rdi
	.p2align 4,,10
	.p2align 3
.L74:
	mov	rax, r12
	shr	r12
	cmp	edx, DWORD PTR [rbx+r12*4]
	jle	.L73
	sub	rax, r12
	lea	rbx, [rbx+rax*4]
.L73:
	test	r12, r12
	jne	.L74
	jmp	.L93
.L85:
	xor	ebx, ebx
	jmp	.L67
	.cfi_endproc
.LFE24:
	.size	array_bsearch, .-array_bsearch
	.section	.rodata
	.align 8
	.type	__PRETTY_FUNCTION__.0, @object
	.size	__PRETTY_FUNCTION__.0, 14
__PRETTY_FUNCTION__.0:
	.string	"array_permute"
	.align 8
	.type	__PRETTY_FUNCTION__.1, @object
	.size	__PRETTY_FUNCTION__.1, 14
__PRETTY_FUNCTION__.1:
	.string	"array_shuffle"
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.LC2:
	.quad	0
	.quad	1
	.quad	2
	.quad	3
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC4:
	.quad	4
	.ident	"GCC: (GNU) 13.2.1 20230801"
	.section	.note.GNU-stack,"",@progbits
