#include "comm_result.h"

std::string comm_result_to_string(comm_result_t input)
{
    switch (input){
        case comm_result_t::ConnectionFailed:           return "Connection failed";
        case comm_result_t::NotConnected:               return "Not connected";
        case comm_result_t::SessionFinalizedAndClosed:  return "Session finalized and closed";
        case comm_result_t::Success:                    return "Success";
        case comm_result_t::Timeout:                    return "Timeout";
        default:                                        return "Unrecognized result";
    }
}