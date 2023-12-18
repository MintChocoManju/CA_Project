.data
    n: .word 10
    
.text
.globl __start

FUNCTION:
    addi s1, x0, 2 # s1 = 2
    addi s2, x0, 6 # s2 = 6
    addi s3, x0, 5 # s3 = 5
T:
    addi sp, sp, -8 # Save return address and n on stack
    sw sp, ra, 4
    sw sp, a0, 0
    bge a0, s1, REC # Recursion if n >= 2
    addi a0, x0, 2 # Base case, T(1) = 2
    addi sp, sp, 8 # No need to restore value, since no other function called
    jalr x0, ra, 0 # Return
REC:
    srai a0, a0, 1 # a0 = n // 2
    jal ra, T # Call T
    lw t1, sp, 0 # Restore n
    lw ra, sp, 4 # Restore return address
    addi sp, sp, 8 # Pop stack
    mul t1, t1, s2 # t1 = n * 6
    mul a0, a0, s3 # a0 = a0 * 5 (= 5T(n//2))
    add a0, a0, t1 # a0 = a0 + t1 (= 5T(n//2) + 6n)
    addi a0, a0, 4 # a0 = a0 + 4 (= 5T(n//2) + 6n + 4)
    jalr x0, ra, 0  # Return
# Do NOT modify this part!!!
__start:
    la   t0, n
    lw   x10, 0(t0)
    jal  x1,FUNCTION
    la   t0, n
    sw   x10, 4(t0)
    addi a0,x0,10
    ecall