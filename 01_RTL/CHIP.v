//----------------------------- DO NOT MODIFY THE I/O INTERFACE!! ------------------------------//
module CHIP #(                                                                                  //
    parameter BIT_W = 32                                                                        //
)(                                                                                              //
    // clock                                                                                    //
        input               i_clk,                                                              //
        input               i_rst_n,                                                            //
    // instruction memory                                                                       //
        input  [BIT_W-1:0]  i_IMEM_data,                                                        //
        output [BIT_W-1:0]  o_IMEM_addr,                                                        //
        output              o_IMEM_cen,                                                         //
    // data memory                                                                              //
        input               i_DMEM_stall,                                                       //
        input  [BIT_W-1:0]  i_DMEM_rdata,                                                       //
        output              o_DMEM_cen,                                                         //
        output              o_DMEM_wen,                                                         //
        output [BIT_W-1:0]  o_DMEM_addr,                                                        //
        output [BIT_W-1:0]  o_DMEM_wdata,                                                       //
    // finnish procedure                                                                        //
        output              o_finish,                                                           //
    // cache                                                                                    //
        input               i_cache_finish,                                                     //
        output              o_proc_finish                                                       //
);                                                                                              //
//----------------------------- DO NOT MODIFY THE I/O INTERFACE!! ------------------------------//

// ------------------------------------------------------------------------------------------------------------------------------------------------------
// Parameters
// ------------------------------------------------------------------------------------------------------------------------------------------------------

    parameter SHSIZE = 5;
    // opcode
    localparam AUIPC  = 7'b0010111;
    localparam JAL    = 7'b1101111;
    localparam JALR   = 7'b1100111;
    localparam ROP32  = 7'b0110011;
    localparam IOP32  = 7'b0010011;
    localparam LOAD   = 7'b0000011;
    localparam STORE  = 7'b0100011;
    localparam BRANCH = 7'b1100011;
    localparam ECALL  = 7'b1110011;
    // funct7 + funct3
    localparam ADD = 10'b0000000_000;
    localparam SUB = 10'b0100000_000;
    localparam SLL = 10'b0000000_001;
    localparam SLT = 10'b0000000_010;
    localparam XOR = 10'b0000000_100;
    localparam SRA = 10'b0100000_101;
    localparam AND = 10'b0000000_111;
    localparam MUL = 10'b0000001_000;
    // branch funct3
    localparam BEQ = 3'b000;
    localparam BNE = 3'b001;
    localparam BLT = 3'b100;
    localparam BGE = 3'b101;
    // JALR mask
    localparam MASK_JALR = {{BIT_W-1{1'b1}}, 1'b0};

// ------------------------------------------------------------------------------------------------------------------------------------------------------
// Wires and Registers
// ------------------------------------------------------------------------------------------------------------------------------------------------------
    
    reg [BIT_W-1:0] PC_r, PC_w;

    reg imem_cen;
    wire [6:0] opcode;
    wire stall, reg_write, has_rs1, has_rs2, has_rd;
    wire [4:0] rs1, rs2, rd;
    wire [BIT_W-1:0] reg_rdata1, reg_rdata2;
    reg  [BIT_W-1:0] reg_wdata;
    wire [BIT_W-1:0] imm_I, imm_S, imm_B, imm_U, imm_J;

    wire alu_active, alu_slt, alu_mul, muldiv_stall;
    reg  branch_taken;
    reg  [9:0] alu_operation;
    reg  [BIT_W-1:0] alu_operand, alu_result;
    wire [2*BIT_W-1:0] alu_product;

// ------------------------------------------------------------------------------------------------------------------------------------------------------
// Continuous Assignment
// ------------------------------------------------------------------------------------------------------------------------------------------------------

    assign o_IMEM_addr   = PC_r;
    assign o_IMEM_cen    = imem_cen;
    assign o_DMEM_cen    = (opcode == LOAD) || (opcode == STORE);
    assign o_DMEM_wen    = (opcode == STORE);
    assign o_DMEM_addr   = o_DMEM_cen ? alu_result : {BIT_W{1'b0}};
    assign o_DMEM_wdata  = (opcode == STORE) ? reg_rdata2 : {BIT_W{1'b0}};
    assign o_finish      = (opcode == ECALL) && !stall;
    assign o_proc_finish = (opcode == ECALL);

    assign stall = i_DMEM_stall | muldiv_stall;

    assign opcode     = i_IMEM_data[6:0];
    assign has_rs1    = (opcode == JALR) || (opcode == BRANCH) || (opcode == LOAD) || (opcode == STORE)
                     || (opcode == IOP32) || (opcode == ROP32);
    assign rs1        = has_rs1 ? i_IMEM_data[19:15] : 5'b0;
    assign has_rs2    = (opcode == BRANCH) || (opcode == STORE) || (opcode == ROP32);
    assign rs2        = has_rs2 ? i_IMEM_data[24:20] : 5'b0;
    assign has_rd     = (opcode == AUIPC) || (opcode == JAL) || (opcode == JALR) || (opcode == LOAD)
                     || (opcode == IOP32) || (opcode == ROP32);
    assign rd         = has_rd ? i_IMEM_data[11:7] : 5'b0;
    assign reg_write  = has_rd && !stall;
    assign imm_I      = {{BIT_W-11{i_IMEM_data[31]}}                , i_IMEM_data[30:20]};
    assign imm_S      = {{BIT_W-11{i_IMEM_data[31]}}                , i_IMEM_data[30:25], i_IMEM_data[11:7]};
    assign imm_B      = {{BIT_W-12{i_IMEM_data[31]}}, i_IMEM_data[7], i_IMEM_data[30:25], i_IMEM_data[11:8], 1'b0};
    assign imm_U      = {i_IMEM_data[31:12], 12'b0};
    assign imm_J      = {{BIT_W-20{i_IMEM_data[31]}}, i_IMEM_data[19:12], i_IMEM_data[20], i_IMEM_data[30:21], 1'b0};
    assign alu_active = (opcode == BRANCH) || (opcode == LOAD) || (opcode == STORE) || (opcode == IOP32) || (opcode == ROP32);
    assign alu_mul    = (alu_operation == MUL);

// ------------------------------------------------------------------------------------------------------------------------------------------------------
// Submoddules
// ------------------------------------------------------------------------------------------------------------------------------------------------------

    Reg_file reg0(               
        .i_clk  (i_clk),             
        .i_rst_n(i_rst_n),         
        .wen    (reg_write),          
        .rs1    (rs1),                
        .rs2    (rs2),                
        .rd     (rd),                 
        .wdata  (reg_wdata),             
        .rdata1 (reg_rdata1),           
        .rdata2 (reg_rdata2)
    );

    SLT_unit#(.BW(BIT_W)) slt_unit_alu(
        .i_a   (reg_rdata1),
        .i_b   (alu_operand),
        .o_slt (alu_slt)
    );

    MULDIV_unit#(.BW(BIT_W)) muldiv_unit_alu(
        .i_clk     (i_clk),
        .i_rst_n   (i_rst_n),
        .i_a       (reg_rdata1),
        .i_b       (alu_operand),
        .i_valid   (alu_mul),
        .o_product (alu_product),
        .o_stall   (muldiv_stall)
    );

// ------------------------------------------------------------------------------------------------------------------------------------------------------
// Always Blocks
// ------------------------------------------------------------------------------------------------------------------------------------------------------

    always @(*) begin
        if (stall) PC_w = PC_r;
        else begin
            case (opcode)
                JAL:     PC_w = PC_r + imm_J;
                JALR:    PC_w = (reg_rdata1 + imm_I) & MASK_JALR;
                BRANCH:  PC_w = PC_r + (branch_taken ? imm_B : 4);
                default: PC_w = PC_r + 4;
            endcase
        end
    end

    always @(*) begin
        if (stall) reg_wdata = {BIT_W{1'b0}};
        else begin
            case (opcode)
                AUIPC:        reg_wdata = PC_r + imm_U;
                JAL, JALR:    reg_wdata = PC_r + 4;
                ROP32, IOP32: reg_wdata = alu_result;
                LOAD:         reg_wdata = i_DMEM_rdata;
                default:      reg_wdata = {BIT_W{1'b0}};
            endcase
        end
    end

    always @(*) begin
        case (opcode)
            ROP32: begin
                alu_operation = {i_IMEM_data[31:25], i_IMEM_data[14:12]};
                alu_operand   = reg_rdata2;
            end
            IOP32: begin
                if (i_IMEM_data[14:12] == 3'b101) alu_operation = {i_IMEM_data[31:25], i_IMEM_data[14:12]}; // Right shift special case
                else                              alu_operation = {7'b0              , i_IMEM_data[14:12]};
                alu_operand   = imm_I;
            end
            LOAD: begin
                alu_operation = ADD;
                alu_operand   = imm_I;
            end
            STORE: begin
                alu_operation = ADD;
                alu_operand   = imm_S;
            end
            BRANCH: begin
                alu_operation = XOR;
                alu_operand   = reg_rdata2;
            end
            default: begin
                alu_operation = 10'b0;
                alu_operand   = {BIT_W{1'b0}};
            end
        endcase
    end

    always @(*) begin
        if (alu_active) begin
            case (alu_operation)
                ADD:     alu_result = reg_rdata1 + alu_operand;
                SUB:     alu_result = reg_rdata1 - alu_operand;
                SLL:     alu_result = reg_rdata1 << alu_operand[SHSIZE-1:0];
                SLT:     alu_result = {{BIT_W-1{1'b0}}, alu_slt};
                XOR:     alu_result = reg_rdata1 ^ alu_operand;
                SRA:     alu_result = $unsigned($signed(reg_rdata1) >>> alu_operand[SHSIZE-1:0]);
                AND:     alu_result = reg_rdata1 & alu_operand;
                MUL:     alu_result = alu_product[BIT_W-1:0];
                default: alu_result = {BIT_W{1'b0}};
            endcase
        end
        else alu_result = {BIT_W{1'b0}};
    end

    always @(*) begin
        if (opcode == BRANCH) begin
            case (i_IMEM_data[14:12])
                BEQ:     branch_taken = ~|alu_result;
                BNE:     branch_taken = |alu_result;
                BLT:     branch_taken = alu_slt;
                BGE:     branch_taken = !alu_slt;
                default: branch_taken = 1'b0;
            endcase
        end
        else begin
            branch_taken = 1'b0;
        end
    end

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            PC_r     <= 32'h00010000; // Do not modify this value!!!
            imem_cen <= 1'b1;
        end
        else begin
            PC_r     <= PC_w;
            imem_cen <= !stall;
        end
    end

endmodule

module Reg_file(i_clk, i_rst_n, wen, rs1, rs2, rd, wdata, rdata1, rdata2);
   
    parameter BITS = 32;
    parameter word_depth = 32;
    parameter addr_width = 5; // 2^addr_width >= word_depth
    
    input i_clk, i_rst_n, wen; // wen: 0:read | 1:write
    input [BITS-1:0] wdata;
    input [addr_width-1:0] rs1, rs2, rd;

    output [BITS-1:0] rdata1, rdata2;

    reg [BITS-1:0] mem [0:word_depth-1];
    reg [BITS-1:0] mem_nxt [0:word_depth-1];

    integer i;

    assign rdata1 = mem[rs1];
    assign rdata2 = mem[rs2];

    always @(*) begin
        for (i=0; i<word_depth; i=i+1)
            mem_nxt[i] = (wen && (rd == i)) ? wdata : mem[i];
    end

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            mem[0] <= 0;
            for (i=1; i<word_depth; i=i+1) begin
                case(i)
                    32'd2: mem[i] <= 32'hbffffff0;
                    32'd3: mem[i] <= 32'h10008000;
                    default: mem[i] <= 32'h0;
                endcase
            end
        end
        else begin
            mem[0] <= 0;
            for (i=1; i<word_depth; i=i+1)
                mem[i] <= mem_nxt[i];
        end       
    end
endmodule

module MULDIV_unit #(
    parameter BW = 32
) (
    input             i_clk,
    input             i_rst_n,
    input  [BW-1:0]   i_a,
    input  [BW-1:0]   i_b,
    input             i_valid,
    output [2*BW-1:0] o_product,
    output            o_stall
);

    localparam S_IDLE = 1'b0;
    localparam S_CALC = 1'b1;

    reg state_w, state_r;
    reg [BW-1:0]   a_w, a_r, b_w, b_r;
    wire [2*BW-1:0] product[BW:0];
    genvar i;

    assign o_stall   = i_valid && (state_r == S_IDLE);
    assign o_product = (state_r == S_CALC) ? product[BW] : {(2*BW){1'b0}};

    assign product[0] = {{BW{1'b0}}, i_a};
    generate
        for (i = 0; i < BW; i = i + 1)
        begin: gen_product
            assign product[i+1] = (product[i] + (product[i][0] ? {i_b, {BW{1'b0}}} : {(2*BW){1'b0}})) >> 1;
        end
    endgenerate

    always @(*) begin
        case (state_r)
            S_IDLE: begin
                if (i_valid) begin
                    state_w = S_CALC;
                    a_w     = i_a;
                    b_w     = i_b;
                end
                else begin
                    state_w = S_IDLE;
                    a_w     = {BW{1'b0}};
                    b_w     = {BW{1'b0}};
                end
            end
            S_CALC: begin
                state_w = S_IDLE;
                a_w     = {BW{1'b0}};
                b_w     = {BW{1'b0}};
            end
        endcase
    end

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_r   <= S_IDLE;
        end
        else begin
            state_r   <= state_w;
        end
    end


endmodule

module SLT_unit #(
    parameter BW = 32
) (
    input [BW-1:0] i_a,
    input [BW-1:0] i_b,
    output         o_slt
);

    wire [BW-1:0] diff;
    reg slt;

    assign diff = i_a - i_b;
    assign o_slt = slt;

    always @(*) begin
        case ({i_a[BW-1], i_b[BW-1]})
            2'b01:   slt = 1'b0;             // pos, neg
            2'b10:   slt = 1'b1;             // neg, pos
            default: slt = diff[BW-1];
        endcase
    end
    
endmodule

module Cache#(
    parameter BIT_W = 32,
    parameter ADDR_W = 32
) (
    input i_clk,
    input i_rst_n,
    // processor interface
    input i_proc_cen,
    input i_proc_wen,
    input [ADDR_W-1:0] i_proc_addr,
    input [BIT_W-1:0]  i_proc_wdata,
    output [BIT_W-1:0] o_proc_rdata,
    output o_proc_stall,
    input i_proc_finish,
    output o_cache_finish,
    // memory interface
    output o_mem_cen,
    output o_mem_wen,
    output [ADDR_W-1:0] o_mem_addr,
    output [BIT_W*4-1:0]  o_mem_wdata,
    input [BIT_W*4-1:0] i_mem_rdata,
    input i_mem_stall,
    output o_cache_available,
    // others
    input  [ADDR_W-1: 0] i_offset
);

    assign o_cache_available = 1'b1; // change this value to 1 if the cache is implemented

// ------------------------------------------------------------------------------------------------------------------------------------------------------
// Parameters
// ------------------------------------------------------------------------------------------------------------------------------------------------------

    parameter BYOS_W = 2;
    parameter BLOS_W = 2;
    parameter BLKI_W = 4;
    parameter TAG_W  = ADDR_W - BLKI_W - BLOS_W - BYOS_W;
    parameter SHSIZE = 5;
    localparam S_IDLE = 3'd0;
    localparam S_READ = 3'd1;
    localparam S_WRITE = 3'd2;
    localparam S_START = 3'd3;
    localparam S_CLEAN = 3'd4;

// ------------------------------------------------------------------------------------------------------------------------------------------------------
// Wires and Registers
// ------------------------------------------------------------------------------------------------------------------------------------------------------

    reg  [2:0] state_r, state_w;
    reg  [BIT_W*4-1:0] cache_data [{BLKI_W{1'b1}}:0], cache_data_w [{BLKI_W{1'b1}}:0];
    reg  [TAG_W  -1:0] cache_tags [{BLKI_W{1'b1}}:0], cache_tags_w [{BLKI_W{1'b1}}:0];
    reg                cache_valid[{BLKI_W{1'b1}}:0], cache_valid_w[{BLKI_W{1'b1}}:0];
    reg                cache_dirty[{BLKI_W{1'b1}}:0], cache_dirty_w[{BLKI_W{1'b1}}:0];
    
    wire [TAG_W -1:0] tag;
    reg  [TAG_W -1:0] tag_r;
    wire [BLKI_W-1:0] block_i;
    reg  [BLKI_W-1:0] block_i_r;
    wire [BLOS_W-1:0] block_os;
    reg  [BLOS_W-1:0] block_os_r;
    wire [BYOS_W-1:0] byte_os;

    reg                proc_stall, mem_cen, mem_wen;
    reg  [BIT_W  -1:0] proc_rdata, mem_wdata;
    wire [BIT_W*4-1:0] alloc_data_w, alloc_mask_w;
    reg  [BIT_W*4-1:0] alloc_data,   alloc_mask;
    reg  [ADDR_W -1:0] mem_addr;

    wire miss, make_dirty;

// ------------------------------------------------------------------------------------------------------------------------------------------------------
// Continuous Assignment
// ------------------------------------------------------------------------------------------------------------------------------------------------------

    assign o_proc_stall = proc_stall;
    assign o_proc_rdata = proc_rdata;

    assign o_mem_cen   = mem_cen;
    assign o_mem_wen   = mem_wen;
    assign o_mem_addr  = mem_addr;
    assign o_mem_wdata = mem_wdata;

    assign {tag, block_i, block_os, byte_os} = i_proc_addr - i_offset;
    assign miss       = !cache_valid[block_i] || (cache_tags[block_i] != tag);
    assign alloc_data_w = (state_r != S_IDLE) ? alloc_data
                                              : mem_wen ? (i_proc_wdata << (block_os << SHSIZE)) : {(BIT_W*4){1'b0}};
    assign alloc_mask_w = (state_r != S_IDLE) ? alloc_mask
                                              : mem_wen ? ~({BIT_W{1'b1}} << (block_os << SHSIZE)) : {(BIT_W*4){1'b1}};

// ------------------------------------------------------------------------------------------------------------------------------------------------------
// Always Blocks
// ------------------------------------------------------------------------------------------------------------------------------------------------------

    always @(*) begin
        case (state_r)
            S_IDLE: begin
                if (miss && cache_dirty[block_i]) state_w = S_WRITE;
                else if (miss)                    state_w = S_READ;
                else if (i_proc_finish)           state_w = S_CLEAN;
                else                              state_w = S_IDLE;
            end
            S_READ:  state_w = i_mem_stall ? S_READ  : S_IDLE;
            S_WRITE: state_w = i_mem_stall ? S_WRITE : S_IDLE;
            S_START: state_w = S_READ;
            S_CLEAN: state_w = S_CLEAN;
        endcase
    end

    integer i;
    always @(*) begin
        for (i = 0; i <= {BLKI_W{1'b1}}; i = i + 1) begin
            if (state_r == S_IDLE && i_proc_cen && i_proc_wen && !miss && i == block_i) begin
                cache_data_w [i] = cache_data[i] & alloc_mask_w | alloc_data_w;
                cache_tags_w [i] = cache_tags[i];
                cache_valid_w[i] = 1'b1;
            end
            else if (state_r == S_READ && i == block_i_r) begin
                cache_data_w [i] = i_mem_rdata & alloc_mask | alloc_data;
                cache_tags_w [i] = tag_r;
                cache_valid_w[i] = 1'b1;
            end
            else begin
                cache_data_w [i] = cache_data [i];
                cache_tags_w [i] = cache_tags [i];
                cache_valid_w[i] = cache_valid[i];
            end
        end
    end

    always @(*) begin
        for (i = 0; i <= {BLKI_W{1'b1}}; i = i + 1) begin
            if (state_r == S_IDLE && i_proc_cen && i_proc_wen && i == block_i) cache_dirty_w[i] = 1'b1;
            else if (state_r == S_READ && alloc_mask == {(BIT_W*4){1'b1}})     cache_dirty_w[i] = 1'b0;
            else                                                               cache_dirty_w[i] = cache_dirty[i];
        end
    end

    always @(*) begin
        case (state_r)
            S_IDLE:           proc_stall = i_proc_cen && miss;
            S_READ:           proc_stall = i_mem_stall;
            S_WRITE, S_START: proc_stall = 1'b1;
            default:          proc_stall = 1'b0;
        endcase
    end

    always @(*) begin
        case (state_r)
            S_IDLE:  proc_rdata = cache_data  [block_i][{block_os,   {(BYOS_W+3){1'b0}}} +: BIT_W];
            S_READ:  proc_rdata = cache_data_w[block_i][{block_os_r, {(BYOS_W+3){1'b0}}} +: BIT_W];
            default: proc_rdata = {BIT_W{1'b0}};
        endcase
    end

    always @(*) begin
        case (state_r)
            S_IDLE: begin
                mem_cen = i_proc_cen && miss;
                mem_wen = i_proc_cen && miss && cache_dirty[block_i];
            end
            S_READ: begin
                mem_cen = 1'b1;
                mem_wen = 1'b0;
            end
            S_READ, S_START: begin
                mem_cen = 1'b1;
                mem_wen = 1'b0;
            end
            S_WRITE: begin
                mem_cen = 1'b1;
                mem_wen = 1'b1;
            end
            default: begin
                mem_cen = 1'b0;
                mem_wen = 1'b0;
            end
        endcase
    end

    always @(*) begin
        case (state_r)
            S_IDLE:          mem_addr = (cache_dirty[block_i] ? {cache_tags[block_i], block_i, {(BLOS_W+BYOS_W){1'b0}}}
                                                              : {tag,                 block_i, {(BLOS_W+BYOS_W){1'b0}}}) + i_offset;
            S_READ, S_START: mem_addr = {tag_r                , block_i_r, {(BLOS_W+BYOS_W){1'b0}}} + i_offset;
            S_WRITE:         mem_addr = {cache_tags[block_i_r], block_i_r, {(BLOS_W+BYOS_W){1'b0}}} + i_offset;
            default:         mem_addr = {ADDR_W{1'b0}};
        endcase
    end

    always @(*) begin
        case (state_r)
            S_IDLE:  mem_wdata = cache_data[block_i];
            S_WRITE: mem_wdata = cache_data[block_i_r];
            default: mem_wdata = {(BIT_W*4){1'b0}};
        endcase
    end

endmodule