
#include <stdio.h>
#include <iostream>
#include "pin.H"

FILE* trace;

std::vector<std::string> routine_names;
uint64_t current_routine;


VOID record_read_ins(VOID* ip, VOID* addr) {
    fprintf(trace, "%p: R %p\n", ip, addr);
}

VOID record_write_ins(VOID* ip, VOID* addr, UINT32 memop) {
    fprintf(trace, "%p: W %p %d\n", ip, addr, memop);
}

VOID record_enter_rtn(UINT64 routine_id, VOID* addr) {
    fprintf(trace, "%p: Entering %s (%llu)\n", addr, routine_names[routine_id].c_str(), routine_id);
}

VOID record_exit_rtn(UINT64 routine_id, VOID* addr) {
    fprintf(trace, "%p: Exiting %s (%llu)\n", addr, routine_names[routine_id].c_str(), routine_id);
}


VOID instrument_ins(INS ins, VOID* v) {
    // Instruments memory accesses using a predicated call, i.e.
    // the instrumentation is called iff the instruction will actually be executed.

    // On the IA-32 and Intel(R) 64 architectures conditional moves and REP
    // prefixed instructions appear as predicated instructions in Pin.
    UINT32 memOperands = INS_MemoryOperandCount(ins);

    // Iterate over each memory operand of the instruction.
    for (UINT32 memOp = 0; memOp < memOperands; memOp++)
    {
        if (INS_MemoryOperandIsRead(ins, memOp)) {
            INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR) record_read_ins,
                                     IARG_INST_PTR,
                                     IARG_MEMORYOP_EA, memOp,
                                     IARG_END);
        }

        // Note that in some architectures a single memory operand can be
        // both read and written (for instance incl (%eax) on IA-32)
        // In that case we instrument it once for read and once for write.
        if (INS_MemoryOperandIsWritten(ins, memOp)) {
            INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR) record_write_ins,
                                     IARG_INST_PTR,
                                     IARG_MEMORYOP_EA, memOp,
                                     IARG_END);
        }
    }
}

// Called every time a _new_ routine is _executed_
void instrument_rtn(RTN rtn, VOID* v) {

    std::string rtn_name = RTN_Name(rtn);
    ADDRINT rtn_address = RTN_Address(rtn);

    routine_names.push_back(rtn_name);
    current_routine++;

    printf("Instrumenting %s (%llu)\n", rtn_name.c_str(), current_routine);

    RTN_Open(rtn);
    // Insert a call to record_enter_rtn at the routine's entry point
    RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR) record_enter_rtn,
                   IARG_UINT64, current_routine,
                   IARG_ADDRINT, rtn_address,
                   IARG_END);

    // Insert a call to record_exit_rtn at the routine's exit point
    // (Warning: PIN might not find all exit points!)
    RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR) record_exit_rtn,
                   IARG_UINT64, current_routine,
                   IARG_ADDRINT, rtn_address,
                   IARG_END);
    RTN_Close(rtn);
}

VOID instrument_finish(INT32 code, VOID* v) {
    fprintf(trace, "#eof\n");
    fclose(trace);
}

INT32 print_usage() {
    PIN_ERROR("This Pintool prints a trace of memory addresses\n" + KNOB_BASE::StringKnobSummary() + "\n");
    return -1;
}

int main(int argc, char* argv[]) {

    PIN_InitSymbols();
    if (PIN_Init(argc, argv)) return print_usage();

    trace = fopen("pinatrace.out", "w");

    RTN_AddInstrumentFunction(instrument_rtn, 0);
    INS_AddInstrumentFunction(instrument_ins, 0);
    PIN_AddFiniFunction(instrument_finish, 0);

    PIN_StartProgram(); // Never returns
    return 0;
}

