.globl __start
__start:
    la a1, __mem
    lw a0, a1, 0 # Load input n
    addi s1, x0, 2 # s1 = 2
    addi s2, x0, 6 # s2 = 6
    addi s3, x0, 5 # s3 = 5
    jal ra, T # Call function
    sw a1, t0, 4 # Store output
    ecall
T:
    addi sp, sp, -8 # Save return address and n on stack
    sw sp, ra, 4
    sw sp, a0, 0
    bge a0, s1, Tr # Recursion if n >= 2
    addi t0, x0, 2 # Base case, T(1) = 2
    addi sp, sp, 8 # No need to restore value, since no other function called
    jalr x0, ra, 0 # Return
Tr: srai a0, a0, 1 # a0 = n // 2
    jal ra, T # Call T
    lw a0, sp, 0 # Restore n
    lw ra, sp, 4 # Restore return address
    addi sp, sp, 8 # Pop stack
    mul t1, a0, s2 # t1 = n * 6
    mul t0, t0, s3 # t0 = t0 * 5 (= 5T(n//2))
    add t0, t0, t1 # t0 = t0 + t1 (= 5T(n//2) + 6n)
    addi t0, t0, 4 # t0 = t0 * 4 (= 5T(n//2) + 6n + 4)
    jalr x0, ra, 0  # Return
__mem: