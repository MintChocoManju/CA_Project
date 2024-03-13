# RISC-V CPU with Direct-mapped Cache
## Description
This project is an RTL (Register Transfer Level) implementation of a RISC-V CPU featuring a 2kb direct-mapped L1 cache. It aims to demonstrate  the operational principles of CPU stalls during multicycle operations and the process of data access with an L1 cache. A write-back cache policy is adopted to enhance efficiency, particularly for tasks with substantial IO requirements, such as sorting.

## Requirements
- Synopsys VCS (Verilog Compiler Simulator) or compatible tool

## How to Run
To run a simulation with VCS, execute the following command in your terminal:

    vcs /00_TB/tb.v /01_RTL/CHIP.v -full64 -R -debug_access+all +v2k +notimingcheck +I[1,2,3...]

- '+Ix' denotes the test set to be run, corresponding to each assembly file in '02_Assembly'.

## Credits
The source code is developed as part of the NTU EE4039 Computer Architecture course, September 2023. The files '01_RTL/CHIP.v' and '02_Assembly/I2_hw1.s' are original contributions, while the remaining files were provided by the course teaching assistants.
