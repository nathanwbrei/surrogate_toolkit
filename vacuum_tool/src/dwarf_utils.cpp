
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include "dwarf_utils.hpp"
#include <iostream>

void hello_from_vacuum_tool() {
    std::cout << "Vacuum tool" << std::endl;
}

DwarfContext::DwarfContext(std::string path_to_executable) {

    // On MacOS the debug info might live in a .dSYM file which libdwarf chases down for us
    const size_t buffer_length = 1000;
    char true_path_to_executable[buffer_length];
    true_path_to_executable[0] = 0;

    error_code = dwarf_init_path(path_to_executable.c_str(),
                    true_path_to_executable,   // Out parameter to give "true" path to executable
                    buffer_length,
                    DW_GROUPNUMBER_ANY,
                    nullptr,  // No error handling function
                    nullptr,  // No context information given to the nonexistent error handling function
                    &context, // Updated if success
                    &error    // Updated if failure
                );
    if (error_code != DW_DLV_OK) {
        throw std::runtime_error("Unable to create DWARF context for " + path_to_executable);
    }
    std::cout << "Successfully created a DWARF context for " << true_path_to_executable << std::endl;
}

DwarfContext::~DwarfContext() {
    dwarf_finish(context);
}

std::optional<VariableInfo> DwarfContext::identify_global_primitive() {

    int error_code = DW_DLV_OK;
    while (error_code == DW_DLV_OK) {

        Dwarf_Unsigned cu_header_length = 0;
        Dwarf_Half version_stamp = 0;
        Dwarf_Off abbreviation_offset = 0;
        Dwarf_Half address_size = 0;
        Dwarf_Half offset_size = 0;
        Dwarf_Half extension_size = 0;
        Dwarf_Sig8 type_signature; // memsetted later
        Dwarf_Unsigned type_offset = 0;
        Dwarf_Unsigned next_cu_header_offset = 0;
        Dwarf_Half header_cu_type = DW_UT_compile;

        memset(&type_signature,0, sizeof(type_signature));

        error_code = dwarf_next_cu_header_d(
                        context,
                        true,  // is_info
                        &cu_header_length,
                        &version_stamp,
                        &abbreviation_offset,
                        &address_size,
                        &offset_size,
                        &extension_size,
                        &type_signature,
                        &type_offset,
                        &next_cu_header_offset,
                        &header_cu_type,
                        &error
                     );

        if (error_code == DW_DLV_ERROR) {
            const char *error_msg = error ? dwarf_errmsg(error): "Unknown error";
            std::cout << "Found error traversing DWARF compilation unit headers: " <<  error_msg << std::endl;
            break;
        }
        else if (error_code == DW_DLV_NO_ENTRY) {
            // We ran out of headers, so we are done
            break;
        }

        std::cout << "Obtained next_cu_header_offset = " << next_cu_header_offset << std::endl;
        Dwarf_Die cu_die = nullptr;

        error_code = dwarf_siblingof_b(
                         context,
                         nullptr, // Usually parent DIE, but in the case of a CU DIE the library finds it automatically
                                  // based off of hidden state set by the last call to dwarf_next_cu_header_d
                         true,    // is_info
                         &cu_die, // Out parameter in success case
                         &error   // Out parameter in failure case
                     );

        if (error_code == DW_DLV_ERROR) {
            const char *error_msg = error ? dwarf_errmsg(error): "Unknown error";
            std::cout << "Found error traversing DWARF compilation unit DIEs: " <<  error_msg << std::endl;
            break;
        }

        // If we reach this point we now have a CU DIE we can do something with
        std::cout << "Obtained a DIE we can do something with: " << cu_die << std::endl;

        traverse_die(cu_die, 0);

        dwarf_dealloc(context, cu_die, DW_DLA_DIE); // Tell DWARF to deallocate that CU DIE

    }

    return std::nullopt;
}

void DwarfContext::traverse_die(Dwarf_Die parent_die, int level) {

    // do something with that die
    visit_die(parent_die, level);

    // traverse each child
    Dwarf_Die current_child = nullptr;
    int result = dwarf_child(parent_die, &current_child, nullptr);
    if (result == DW_DLV_ERROR) {
        std::cout << "Error finding child" << std::endl;
        exit(1);
    }
    else if (result == DW_DLV_NO_ENTRY) {
        return;
    }
    // At least one child
    traverse_die(current_child, level+1);

    // Look for siblings
    while (true) {
        result = dwarf_siblingof_b(context,
                                   current_child,
                                   true, // is_info
                                   &current_child,
                                   nullptr // errp
                                  );
        if (result == DW_DLV_ERROR) {
            std::cout << "Error finding child-sibling" << std::endl;
            exit(1);
        }
        if (result == DW_DLV_NO_ENTRY) {
            // No more siblings
            break;
        }
        traverse_die(current_child, level+1);
    }
}

void DwarfContext::visit_die(Dwarf_Die die, int level) {
    Dwarf_Half tag = 0;
    const char* tag_name = nullptr;
    int result = dwarf_tag(die, &tag, nullptr);
    if (result != DW_DLV_OK) {
        std::cout << "Unable to read tag" << std::endl;
        exit(1);
    }

    result = dwarf_get_TAG_name(tag, &tag_name);
    if (result != DW_DLV_OK) {
        std::cout << "Unable to stringify tag" << std::endl;
        exit(1);
    }

    char* name = nullptr;
    result = dwarf_diename(die,&name,nullptr);
    if (result == DW_DLV_ERROR) {
        std::cout << "Unable to read diename" << std::endl;
        exit(1);
    }
    else if (result == DW_DLV_NO_ENTRY) {
        name = (char*) "<no DW_AT_name attr>";
    }

    for (int i=0; i<level; ++i) {
        std::cout << "  ";
    }
    std::cout << tag_name << " " << name << std::endl;
}

