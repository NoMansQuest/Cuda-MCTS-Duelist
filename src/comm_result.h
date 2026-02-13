#ifndef _COMM_RESULT_H__
#define _COMM_RESULT_H__

/// @brief Enum represents the outcome of action performed on the TCP socket operations.
enum class comm_result_t : int{

    /// @brief Operation was successful
    Success,

    /// @brief Cannot perform operation as we're not connected to the remote party.
    NotConnected,

    /// @brief Operation encountered a timeout.
    Timeout,

    /// @brief Try to perform the action but connection failed (and we're now disconnected).
    ConnectionFailed,

    /// @brief Remote finalized and closed the session.
    SessionFinalizedAndClosed
};

/// @brief Convert 'comm_result_t' to string.
/// @param input Value to convert.
/// @return String representing the enum value.
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

#endif
