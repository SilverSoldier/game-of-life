#include <gtest/gtest.h>
#include "gol.c" // Include your source file

// Test case for a specific function
TEST(GameOfLifeTest, MoveNextStepTest) {
    int rows = 5;
    int columns = 5;
    int* board = (int*)calloc(rows * columns, sizeof(int));

    // Initialize your board as needed
    // You can set up a specific test case here

    move_next_step(rows, columns, board);

    // Write assertions to check the expected results
    // For example:
    // ASSERT_EQ(expected_value, board[specific_index]);

    free(board);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
