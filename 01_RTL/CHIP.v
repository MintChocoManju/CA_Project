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
    
    // Todo: any combinational/sequential circuit

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
    )(
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

    assign o_cache_available = 1; // change this value to 1 if the cache is implemented

    //------------------------------------------//
    //          default connection              //
    // assign o_mem_cen = i_proc_cen;              //
    // assign o_mem_wen = i_proc_wen;              //
    // assign o_mem_addr = i_proc_addr;            //
    // assign o_mem_wdata = i_proc_wdata;          //
    // assign o_proc_rdata = i_mem_rdata[0+:BIT_W];//
    // assign o_proc_stall = i_mem_stall;          //
    //------------------------------------------//

    // Todo: BONUS
    // Cache state
    parameter S_IDLE = 3'd0;
    parameter S_WRITE = 3'd1;
    parameter S_READ = 3'd2;
    parameter S_WB = 3'd3;
    // parameter S_WT = 3'd4;
    parameter S_WMEM = 3'd5;
    parameter S_RMEM = 3'd6;

    // Cache parameter
    parameter NUM_SETS = 2;
    parameter NUM_LINES = 8;
    parameter LINE_SIZE = 16;   // size of cache line(byte)

    // Calculate bit field parameters
    parameter OFFSET_BITS = $clog2(LINE_SIZE);
    parameter INDEX_BITS = $clog2(NUM_LINES);
    parameter TAG_BITS = 32 - OFFSET_BITS - INDEX_BITS;

    integer x, y;

// ------------------------------------------------------------------------------------------------------------------------------------------------------
// Wires and Registers
// ------------------------------------------------------------------------------------------------------------------------------------------------------
    
    reg [2:0] state, state_nxt;

    //  Cache memory arrays
    reg [TAG_BITS-1:0] cache_tag[0:NUM_LINES-1][0:NUM_SETS-1];
    reg [TAG_BITS-1:0] cache_tag_nxt[0:NUM_LINES-1][0:NUM_SETS-1];
    reg [LINE_SIZE*8-1:0] cache_data[0:NUM_LINES-1][0:NUM_SETS-1];
    reg [LINE_SIZE*8-1:0] cache_data_nxt[0:NUM_LINES-1][0:NUM_SETS-1];
    reg cache_valid[0:NUM_LINES-1][0:NUM_SETS-1];
    reg cache_valid_nxt[0:NUM_LINES-1][0:NUM_SETS-1];
    reg cache_dirty[0:NUM_LINES-1][0:NUM_SETS-1];
    reg cache_dirty_nxt[0:NUM_LINES-1][0:NUM_SETS-1];

    // wire and assignment
    reg [OFFSET_BITS-1:0] i_offset_bit, i_offset_bit_nxt;
    reg [INDEX_BITS-1:0] i_index, i_index_nxt;
    reg [TAG_BITS-1:0] i_tag, i_tag_nxt;
    wire hit_s0, hit_s1;
    wire hit;

    reg [ADDR_W-1:0] i_p_addr, i_p_addr_nxt;
    reg [ADDR_W-1:0] i_p_wdata, i_p_wdata_nxt;

    reg [BIT_W-1:0] o_p_data, o_p_data_nxt;
    reg [BIT_W-1:0] o_m_addr, o_m_addr_nxt;
    reg [BIT_W*4-1:0] o_m_data, o_m_data_nxt;

    reg o_cen, o_wen;
    reg o_cen_nxt, o_wen_nxt;
    reg o_finish;

    reg [1:0] write_bit;

    reg cache_stall, cache_stall_nxt;

    reg finish;

// ------------------------------------------------------------------------------------------------------------------------------------------------------
// Continuous Assignment
// ------------------------------------------------------------------------------------------------------------------------------------------------------

    assign o_mem_addr = o_m_addr;
    assign o_mem_wdata = o_m_data;
    assign o_mem_wen = o_wen;
    assign o_mem_cen = o_cen;
    assign o_proc_finish = o_finish;
    assign o_proc_rdata = o_p_data_nxt;

    assign o_proc_stall = i_mem_stall | cache_stall_nxt;

    assign o_cache_finish = finish;

    // check hit status
    assign hit_s0 = (cache_valid[i_index][0] && (cache_tag[i_index][0] == i_tag)) ? 1 : 0;
    assign hit_s1 = (cache_valid[i_index][1] && (cache_tag[i_index][1] == i_tag)) ? 1 : 0;
    assign hit = hit_s0 | hit_s1;

// ------------------------------------------------------------------------------------------------------------------------------------------------------
// Always Blocks
// ------------------------------------------------------------------------------------------------------------------------------------------------------

    // update tag, index, offset bits, processor input address
    always @(*) begin
        if(i_proc_cen) begin
            i_offset_bit_nxt = i_proc_addr[OFFSET_BITS-1:0];
            i_index_nxt = i_proc_addr[OFFSET_BITS+INDEX_BITS-1:OFFSET_BITS];
            i_tag_nxt = i_proc_addr[31:OFFSET_BITS+INDEX_BITS];
            i_p_addr_nxt = i_proc_addr;
            i_p_wdata_nxt = i_proc_wdata;
        end
        else begin
            i_offset_bit_nxt = i_offset_bit;
            i_index_nxt = i_index;
            i_tag_nxt = i_tag;
            i_p_addr_nxt = i_p_addr;
            i_p_wdata_nxt = i_p_wdata;
        end
    end


    // FSM
    always @(*) begin
        state_nxt = state;
        o_cen_nxt = o_cen;
        o_wen_nxt = o_wen;
        o_p_data_nxt = o_p_data;
        o_m_data_nxt = o_m_data;
        o_m_addr_nxt = o_m_addr;
        cache_stall_nxt = cache_stall; 
        for (x = 0; x < NUM_LINES; x=x+1) begin
            for (y = 0; y < NUM_SETS; y=y+1) begin
                cache_data_nxt[x][y] = cache_data[x][y];
                cache_tag_nxt[x][y] = cache_tag[x][y];
                cache_dirty_nxt[x][y] = cache_dirty[x][y];
                cache_valid_nxt[x][y] = cache_valid[x][y];
            end
        end
        case(state)
            S_IDLE: begin
                if (i_proc_cen) begin
                    if(i_proc_wen) begin
                        cache_stall_nxt = 1'b1;
                        state_nxt = S_WRITE;
                    end
                    else begin
                        cache_stall_nxt = 1'b1;
                        state_nxt = S_READ;
                    end
                end
                else begin
                    cache_stall_nxt = 1'b0;
                    state_nxt = S_IDLE;
                end
            end
            S_WRITE: begin
                if(hit) begin
                    if(hit_s0) begin
                        case(i_offset_bit)
                            4'h0: begin
                                cache_data_nxt[i_index][0][31:0] = i_p_wdata;
                                cache_dirty_nxt[i_index][0] = 1'b1;
                            end
                            4'h4: begin
                                cache_data_nxt[i_index][0][63:32] = i_p_wdata;
                                cache_dirty_nxt[i_index][0] = 1'b1;
                            end
                            4'h8: begin
                                cache_data_nxt[i_index][0][95:64] = i_p_wdata;
                                cache_dirty_nxt[i_index][0] = 1'b1;
                            end
                            4'hc: begin
                                cache_data_nxt[i_index][0][127:96] = i_p_wdata;
                                cache_dirty_nxt[i_index][0] = 1'b1;
                            end
                            default: begin
                                cache_data_nxt[i_index][0] = cache_data[i_index][0];
                                cache_dirty_nxt[i_index][0] = cache_dirty[i_index][0];
                            end
                        endcase
                        o_m_data_nxt = cache_data_nxt[i_index][0][127:0];
                    end
                    else begin
                        case(i_offset_bit)
                            4'h0: begin
                                cache_data_nxt[i_index][1][31:0] = i_p_wdata;
                                cache_dirty_nxt[i_index][1] = 1'b1;
                            end
                            4'h4: begin
                                cache_data_nxt[i_index][1][63:32] = i_p_wdata;
                                cache_dirty_nxt[i_index][1] = 1'b1;
                            end
                            4'h8: begin
                                cache_data_nxt[i_index][1][95:64] = i_p_wdata;
                                cache_dirty_nxt[i_index][1] = 1'b1;
                            end
                            4'hc: begin
                                cache_data_nxt[i_index][1][127:96] = i_p_wdata;
                                cache_dirty_nxt[i_index][1] = 1'b1;
                            end
                            default: begin
                                cache_data_nxt[i_index][1] = cache_data[i_index][1];
                                cache_dirty_nxt[i_index][1] = cache_dirty[i_index][1];
                            end
                        endcase
                        o_m_data_nxt = cache_data_nxt[i_index][1][127:0];
                    end

                    if({i_p_addr[31:4], 4'h0} < i_offset) begin
                        o_m_addr_nxt = i_offset;
                    end
                    else begin
                        o_m_addr_nxt = {i_p_addr[31:4], 4'h0};
                    end
                    o_wen_nxt = 1'b1;
                    o_cen_nxt = 1'b1;
                    cache_stall_nxt = 1'b1;
                    state_nxt = S_WMEM;
                end
                else begin      // cache miss
                    if (!cache_valid[i_index][0] || !cache_valid[i_index][1]) begin     // set0 or set1 was empty
                        if({i_p_addr[31:4], 4'h0} < i_offset) begin
                            o_m_addr_nxt = i_offset;
                        end
                        else begin
                            o_m_addr_nxt = {i_p_addr[31:4], 4'h0};
                        end
                        state_nxt = S_RMEM;
                    end
                    else begin      // all set are full
                        if({i_p_addr[31:4], 4'h0} < i_offset) begin
                            o_m_addr_nxt = i_offset;
                        end
                        else begin
                            o_m_addr_nxt = {i_p_addr[31:4], 4'h0};
                        end
                        state_nxt = S_RMEM;
                    end
                    o_wen_nxt = 1'b0;
                    o_cen_nxt = 1'b1;
                    o_m_data_nxt = o_m_data;
                    cache_data_nxt[i_index][0] = cache_data[i_index][0];
                    cache_dirty_nxt[i_index][0] = cache_dirty[i_index][0];
                    cache_data_nxt[i_index][1] = cache_data[i_index][1];
                    cache_dirty_nxt[i_index][1] = cache_dirty[i_index][1];
                    cache_stall_nxt = cache_stall;
                end
            end
            S_READ: begin
                if(hit) begin
                    if(hit_s0) begin
                        case(i_offset_bit)
                            4'h0:   o_p_data_nxt = cache_data[i_index][0][31:0];
                            4'h4:   o_p_data_nxt = cache_data[i_index][0][63:32];
                            4'h8:   o_p_data_nxt = cache_data[i_index][0][95:64];
                            4'hc:   o_p_data_nxt = cache_data[i_index][0][127:96];
                            default: o_p_data_nxt = o_p_data;
                        endcase
                    end
                    else begin
                        case(i_offset_bit)
                            4'h0:   o_p_data_nxt = cache_data[i_index][1][31:0];
                            4'h4:   o_p_data_nxt = cache_data[i_index][1][63:32];
                            4'h8:   o_p_data_nxt = cache_data[i_index][1][95:64];
                            4'hc:   o_p_data_nxt = cache_data[i_index][1][127:96];
                            default: o_p_data_nxt = o_p_data;
                        endcase
                    end
                    o_m_addr_nxt = o_m_addr;
                    cache_stall_nxt = 1'b0;
                    o_wen_nxt = o_wen;
                    o_cen_nxt = o_cen;
                    state_nxt = S_IDLE;
                end
                else begin
                    if({i_p_addr[31:4], 4'h0} < i_offset) begin
                        o_m_addr_nxt = i_offset;
                    end
                    else begin
                        o_m_addr_nxt = {i_p_addr[31:4], 4'h0};
                    end
                    cache_stall_nxt = 1'b1;
                    o_wen_nxt = 1'b0;
                    o_cen_nxt = 1'b1;
                    state_nxt = S_RMEM;
                end
            end
            S_WMEM: begin
                if(!i_mem_stall) begin
                    cache_stall_nxt = 1'b0;
                    o_wen_nxt = 1'b0;
                    o_cen_nxt = 1'b0;
                    state_nxt = S_IDLE; 
                end
                else begin
                    cache_stall_nxt = 1'b1;
                    o_wen_nxt = o_wen;
                    o_cen_nxt = o_cen;
                    state_nxt = state;
                end
            end
            S_RMEM: begin
                if(!i_mem_stall) begin      // memory read finish

                    if (!cache_valid[i_index][0]) begin     // set0 is empty
                        if({i_p_addr[31:4], 4'h0} < i_offset) begin      // data address < offset
                            case(i_offset[3:0])
                                4'h0:   cache_data_nxt[i_index][0][BIT_W*4-1:0] = i_mem_rdata;
                                4'h4:   cache_data_nxt[i_index][0][BIT_W*4-1:0] = {i_mem_rdata[95:0], {32{1'b0}}};
                                4'h8:   cache_data_nxt[i_index][0][BIT_W*4-1:0] = {i_mem_rdata[63:0], {64{1'b0}}};
                                4'hc:   cache_data_nxt[i_index][0][BIT_W*4-1:0] = {i_mem_rdata[31:0], {96{1'b0}}};
                                default:    cache_data_nxt[i_index][0][BIT_W*4-1:0] = cache_data[i_index][0][BIT_W*4-1:0];
                            endcase
                        end
                        else begin
                            cache_data_nxt[i_index][0][BIT_W*4-1:0] = i_mem_rdata;
                        end
                        
                        case(i_offset_bit[3:0])
                            4'h0:   o_p_data_nxt = cache_data_nxt[i_index][0][31:0];
                            4'h4:   o_p_data_nxt = cache_data_nxt[i_index][0][63:32];
                            4'h8:   o_p_data_nxt = cache_data_nxt[i_index][0][95:64];
                            4'hc:   o_p_data_nxt = cache_data_nxt[i_index][0][127:96];
                            default: o_p_data_nxt = o_p_data;
                        endcase

                        cache_valid_nxt[i_index][0] = 1'b1;
                        cache_tag_nxt[i_index][0][TAG_BITS-1:0] = i_tag;
                    end
                    else if (!cache_valid[i_index][1]) begin    // set1 is empty
                        if({i_p_addr[31:4], 4'h0} < i_offset) begin
                            case(i_offset[3:0])
                                4'h0:   cache_data_nxt[i_index][1][BIT_W*4-1:0] = i_mem_rdata;
                                4'h4:   cache_data_nxt[i_index][1][BIT_W*4-1:0] = {i_mem_rdata[95:0], {32{1'b0}}};
                                4'h8:   cache_data_nxt[i_index][1][BIT_W*4-1:0] = {i_mem_rdata[63:0], {64{1'b0}}};
                                4'hc:   cache_data_nxt[i_index][1][BIT_W*4-1:0] = {i_mem_rdata[31:0], {96{1'b0}}};
                                default:    cache_data_nxt[i_index][1][BIT_W*4-1:0] = cache_data[i_index][1][BIT_W*4-1:0];
                            endcase
                        end
                        else begin
                            cache_data_nxt[i_index][1][BIT_W*4-1:0] = i_mem_rdata;
                        end
                        
                        case(i_offset_bit[3:0])
                            4'h0:   o_p_data_nxt = cache_data_nxt[i_index][1][31:0];
                            4'h4:   o_p_data_nxt = cache_data_nxt[i_index][1][63:32];
                            4'h8:   o_p_data_nxt = cache_data_nxt[i_index][1][95:64];
                            4'hc:   o_p_data_nxt = cache_data_nxt[i_index][1][127:96];
                            default: o_p_data_nxt = o_p_data;
                        endcase

                        cache_valid_nxt[i_index][1] = 1'b1;
                        cache_tag_nxt[i_index][1][TAG_BITS-1:0] = i_tag;
                    end
                    else begin
                        if({i_p_addr[31:4], 4'h0} < i_offset) begin
                            case(i_offset[3:0])
                                4'h0:   cache_data_nxt[i_index][write_bit][BIT_W*4-1:0] = i_mem_rdata;
                                4'h4:   cache_data_nxt[i_index][write_bit][BIT_W*4-1:0] = {i_mem_rdata[95:0], {32{1'b0}}};
                                4'h8:   cache_data_nxt[i_index][write_bit][BIT_W*4-1:0] = {i_mem_rdata[63:0], {64{1'b0}}};
                                4'hc:   cache_data_nxt[i_index][write_bit][BIT_W*4-1:0] = {i_mem_rdata[31:0], {96{1'b0}}};
                                default:    cache_data_nxt[i_index][write_bit][BIT_W*4-1:0] = cache_data[i_index][write_bit][BIT_W*4-1:0];
                            endcase
                        end
                        else begin
                            cache_data_nxt[i_index][write_bit][BIT_W*4-1:0] = i_mem_rdata;
                        end
                        
                        case(i_offset_bit[3:0])
                            4'h0:   o_p_data_nxt = cache_data_nxt[i_index][write_bit][31:0];
                            4'h4:   o_p_data_nxt = cache_data_nxt[i_index][write_bit][63:32];
                            4'h8:   o_p_data_nxt = cache_data_nxt[i_index][write_bit][95:64];
                            4'hc:   o_p_data_nxt = cache_data_nxt[i_index][write_bit][127:96];
                            default: o_p_data_nxt = o_p_data;
                        endcase

                        cache_valid_nxt[i_index][write_bit] = 1'b1;
                        cache_tag_nxt[i_index][write_bit][TAG_BITS-1:0] = i_tag;
                    end               

                    if(!i_proc_wen) begin    // read
                        cache_stall_nxt = 1'b0;
                        o_cen_nxt = 1'b0;
                        o_wen_nxt = 1'b0;
                        state_nxt = S_IDLE;
                    end
                    else begin      // write
                        cache_stall_nxt = 1'b1;
                        o_cen_nxt = o_cen;
                        o_wen_nxt = o_wen;
                        state_nxt = S_WRITE;
                    end
                end
                else begin
                    cache_stall_nxt = 1'b1;
                    state_nxt = state;
                    o_cen_nxt = o_cen;
                    o_wen_nxt = o_wen;
                    for (x = 0; x < NUM_LINES; x=x+1) begin
                        for (y = 0; y < NUM_SETS; y=y+1) begin
                            cache_data_nxt[x][y] = cache_data[x][y];
                            cache_tag_nxt[x][y] = cache_tag[x][y];
                            cache_dirty_nxt[x][y] = cache_dirty[x][y];
                            cache_valid_nxt[x][y] = cache_valid[x][y];
                        end
                    end
                end
            end
            default: begin
                state_nxt = state;
                o_cen_nxt = o_cen;
                o_wen_nxt = o_wen;
                o_p_data_nxt = o_p_data;
                o_m_data_nxt = o_m_data;
                o_m_addr_nxt = o_m_addr;
                cache_stall_nxt = cache_stall; 
                for (x = 0; x < NUM_LINES; x=x+1) begin
                    for (y = 0; y < NUM_SETS; y=y+1) begin
                        cache_data_nxt[x][y] = cache_data[x][y];
                        cache_tag_nxt[x][y] = cache_tag[x][y];
                        cache_dirty_nxt[x][y] = cache_dirty[x][y];
                        cache_valid_nxt[x][y] = cache_valid[x][y];
                    end
                end
            end
        endcase 
    end
    always @(*) begin
        if (!o_proc_stall) begin
            if (i_proc_finish) begin
                finish = 1'b1;
            end
            else begin
                finish = 1'b0;
            end
        end
        else begin
            finish = 1'b0;
        end
    end
    

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state <= S_IDLE;
            i_offset_bit <= 0;
            i_index <= 0;
            i_tag <= 0;
            i_p_addr <= 0;
            i_p_wdata <= 0;
            for (x = 0; x < NUM_LINES; x=x+1) begin
                for (y = 0; y < NUM_SETS; y=y+1) begin
                    cache_data[x][y] <= 128'b0;
                    cache_tag[x][y] <= {TAG_BITS{1'b0}};
                    cache_dirty[x][y] <= 1'b0;
                    cache_valid[x][y] <= 1'b0;
                end
            end
            cache_stall <= 0;
            o_finish <= 0;
            o_m_addr <= 0;
            o_m_data <= 0;
            o_cen <=0;
            o_wen <= 0;
            o_p_data <= 0;
            write_bit <= 0;
        end
        else begin
            write_bit <= write_bit + 1'b1;
            state <= state_nxt;
            i_offset_bit <= i_offset_bit_nxt;
            i_index <= i_index_nxt;
            i_tag <= i_tag_nxt;
            i_p_addr <= i_p_addr_nxt;
            i_p_wdata <= i_p_wdata_nxt;
            for (x = 0; x < NUM_LINES; x=x+1) begin
                for (y = 0; y < NUM_SETS; y=y+1) begin
                    cache_data[x][y] <= cache_data_nxt[x][y];
                    cache_tag[x][y] <= cache_tag_nxt[x][y];
                    cache_dirty[x][y] <= cache_dirty_nxt[x][y];
                    cache_valid[x][y] <= cache_valid_nxt[x][y];
                end
            end
            cache_stall <= cache_stall_nxt;
            o_p_data <= o_p_data_nxt;
            o_m_data <= o_m_data_nxt;
            o_m_addr <= o_m_addr_nxt;
            o_wen <= o_wen_nxt;
            o_cen <= o_cen_nxt;
        end
    end

endmodule