#ifndef _SESSION_MESSAGE_H__
#define _SESSION_MESSAGE_H__

#include <array>
#include <stdexcept>

/// @brief Enum reflecting the state of the game.
enum class game_state_t : int
{
    /// @brief Game is ongoing
    Playing,
    
    /// @brief Player has won the game.
    PlayerHasWon,

    /// @brief Tie detected (no win, no loss)
    TieDetected
};


/// @brief Message exchanged between participants.
/// @note Is essentially contains the row and column played by the opponent.
struct session_message_t
{
    /// @brief Row allocated by the emitter.
    int allocated_row;

    /// @brief Column allocated by the emitter.
    int allocated_column;

    /// @brief State of the game.
    game_state_t game_state;

    /// @brief Serializes the structure into a 3-byte message
    /// @return Serialized data in an array of 3
    std::array<int, 3> Serialize() const noexcept
    {
        return { allocated_row, allocated_column, static_cast<int>(game_state) };
    }

    /// @brief conversion operator: allows implicit/explicit conversion to std::array<int,3>
    operator std::array<int, 3>() const noexcept
    {
        return this->Serialize();
    }

    /// @brief Deserializes data and produces a valid session_t if possible.
    /// @param input_data Data to deserialize
    /// @exception std::out_of_range exception can be raised in row or column or game_state is not valid.
    /// @return A deserialized session_message_t object.
    static session_message_t Deserialize(std::array<int, 3> input_data)
    {
        const int row = input_data[0];
        const int col = input_data[1];
        const int state  = input_data[2];

        // Validate ranges (adjust if your game uses different bounds)
        if (row < 0 || row > 5) {
            throw std::out_of_range("Deserialize: allocated_row out of range (expected 0..5)");
        }
        if (col < 0 || col > 6) {
            throw std::out_of_range("Deserialize: allocated_column out of range (expected 0..6)");
        }
        if (state < static_cast<int>(game_state_t::Playing) ||
            state > static_cast<int>(game_state_t::TieDetected)) {
            throw std::out_of_range("Deserialize: game_state out of range");
        }

        session_message_t msg;
        msg.allocated_row = row;
        msg.allocated_column = col;
        msg.game_state = static_cast<game_state_t>(state);
        return msg;
    }
};


#endif