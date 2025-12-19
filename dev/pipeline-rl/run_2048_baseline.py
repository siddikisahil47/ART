import asyncio
import math
import random
import string
import xml.etree.ElementTree as ET
from typing import Literal, TypedDict

import requests
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

import art
from art.local.backend import LocalBackend

load_dotenv()

WINNING_VALUE = 64


# Class that keeps track of state for a single game of 2048
class TwentyFortyEightGame(TypedDict):
    id: str
    board: list[list[int | None]]


# Randomly populates a cell on the board with a 2 or 4
def populate_random_cell(game: TwentyFortyEightGame) -> None:
    all_clear_coordinates = [
        (i, j)
        for i in range(len(game["board"]))
        for j in range(len(game["board"][i]))
        if game["board"][i][j] is None
    ]
    random_clear_coordinates = random.choice(all_clear_coordinates)
    # 90% chance to populate a 2, 10% chance to populate a 4
    game["board"][random_clear_coordinates[0]][random_clear_coordinates[1]] = (
        2 if random.random() < 0.9 else 4
    )


# Generates a new game of 2048
def generate_game(board_length: int = 4) -> TwentyFortyEightGame:
    # random 6 character string
    id = "".join(random.choices(string.ascii_letters + string.digits, k=6))
    game = {
        "id": id,
        "board": [[None for _ in range(board_length)] for _ in range(board_length)],
    }

    # populate two random cells
    populate_random_cell(game)
    populate_random_cell(game)

    return game


# Renders the board in a human-readable format
def render_board(game: TwentyFortyEightGame) -> str:
    board = game["board"]
    # print something like this:
    # _    | 2    | _    | 4
    # 4    | 8    | 2    | 16
    # 16   | 32   | 64   | 128
    # _    | 2    | 2    | 4
    # where _ is an empty cell

    max_cell_width = max(
        [len(str(cell)) for row in board for cell in row if cell is not None]
    )

    board_str = ""
    for row in board:
        # pad the cells with spaces to make them the same width
        board_str += "|".join(
            [
                str(cell).rjust(max_cell_width)
                if cell is not None
                else "_".rjust(max_cell_width)
                for cell in row
            ]
        )
        board_str += "\n"
    return board_str


# condense, privileging matches at the start of the sequence
# sequences should be passed starting with cells that are the furthest in the direction in which the board is being condensed
def condense_sequence(sequence: list[int | None]) -> list[int | None]:
    condensed_sequence = []

    gapless_sequence = [cell for cell in sequence if cell is not None]

    i = 0
    while i < len(gapless_sequence):
        if (
            i + 1 < len(gapless_sequence)
            and gapless_sequence[i] == gapless_sequence[i + 1]
        ):
            condensed_sequence.append(gapless_sequence[i] * 2)
            i += 2
        else:
            condensed_sequence.append(gapless_sequence[i])
            i += 1

    # pad the sequence with None at the end
    return condensed_sequence + [None] * (4 - len(condensed_sequence))


# Condenses the board in a given direction
def condense_board(
    game: TwentyFortyEightGame, direction: Literal["left", "right", "up", "down"]
) -> None:
    if direction == "left":
        for row in game["board"]:
            condensed_row = condense_sequence(row)
            for i in range(len(row)):
                row[i] = condensed_row[i]

    if direction == "right":
        for row in game["board"]:
            reversed_row = row[::-1]
            # reverse the row before and after condensing
            condensed_row = condense_sequence(reversed_row)[::-1]
            for i in range(len(row)):
                row[i] = condensed_row[i]

    if direction == "up":
        for col_index in range(len(game["board"][0])):
            column = [row[col_index] for row in game["board"]]

            condensed_column = condense_sequence(column)
            for row_index in range(len(column)):
                game["board"][row_index][col_index] = condensed_column[row_index]

    if direction == "down":
        for col_index in range(len(game["board"][0])):
            column = [row[col_index] for row in game["board"]]
            reversed_column = column[::-1]
            condensed_column = condense_sequence(reversed_column)[::-1]
            for row_index in range(len(column)):
                game["board"][row_index][col_index] = condensed_column[row_index]


# Applies an agent move to the game board
def apply_agent_move(game: TwentyFortyEightGame, move_xml: str) -> None:
    direction = None
    # parse the move
    try:
        root = ET.fromstring(move_xml)
        direction = root.text
    except Exception:
        raise ValueError("Invalid xml")

    if direction not in ["left", "right", "up", "down"]:
        raise ValueError("Invalid direction")

    condense_board(game, direction)

    populate_random_cell(game)


# Returns the maximum cell value on the board
def max_cell_value(game: TwentyFortyEightGame) -> int:
    return max([cell for row in game["board"] for cell in row if cell is not None])


# Returns True if the game is finished
def check_game_finished(game: TwentyFortyEightGame) -> bool:
    if max_cell_value(game) >= WINNING_VALUE:
        return True

    # check if any cell is empty
    if any(cell is None for row in game["board"] for cell in row):
        return False

    return True


# Returns the sum of all the cell values on the board
def total_board_value(game: TwentyFortyEightGame) -> int:
    return sum([cell for row in game["board"] for cell in row if cell is not None])


class Scenario2048(BaseModel):
    step: int


@art.retry(exceptions=(requests.ReadTimeout))
async def rollout(model: art.Model, scenario: Scenario2048) -> art.Trajectory:
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    game = generate_game()

    move_number = 0

    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": "You are an excellent 2048 player. Always choose the move most likely to lead to combine cells to eventually reach the number 2048. Optional moves are 'left', 'right', 'up', 'down'. Return your move as an XML object with a single property 'move', like so: <move>left</move>",
            }
        ],
        metadata={
            "game_id": game["id"],
            "notebook-id": "2048",
            "step": scenario.step,
        },
        reward=0,
    )

    while True:
        trajectory.messages_and_choices.append(
            {"role": "user", "content": render_board(game)}
        )

        try:
            messages = trajectory.messages()
            chat_completion = await client.chat.completions.create(
                max_completion_tokens=128,
                messages=messages,
                model=model.get_inference_name(),
            )
        except Exception as e:
            print("caught exception generating chat completion", e)
            raise e

        choice = chat_completion.choices[0]
        content = choice.message.content
        assert isinstance(content, str)
        trajectory.messages_and_choices.append(choice)

        try:
            apply_agent_move(game, content)
            move_number += 1
        except ValueError:
            trajectory.reward = -1
            break

        if check_game_finished(game):
            max_value = max_cell_value(game)
            board_value = total_board_value(game)
            trajectory.metrics["max_value"] = max_value
            trajectory.metrics["board_value"] = board_value
            trajectory.metrics["move_number"] = move_number

            # try to get as close to the winning value as possible
            # otherwise, try to maximize number of high cells on board
            # but above all else: WIN THE GAME!
            if max_value < WINNING_VALUE:
                # scale max value logarithmically between 0 for 2 and 1 for WINNING_VALUE
                max_value_reward = (math.log(max_value, 2) - 1) / (
                    math.log(WINNING_VALUE, 2) - 1
                )
                # scale board value logarithmically between 0 for 2 * 16 and 1 for WINNING_VALUE * 16
                board_value_reward = (math.log(board_value, 2) - 1) / (
                    math.log(WINNING_VALUE * 16, 2) - 1
                )
                # combine the two rewards, with max value having a higher weight
                trajectory.reward = max_value_reward + (board_value_reward * 0.2)
            else:
                # double reward if the agent wins
                trajectory.reward = 2
            break

    return trajectory


async def main():
    load_dotenv()

    random.seed(42)

    # Declare the model
    model = art.TrainableModel(
        name="agent-001",
        project="2048",
        base_model="OpenPipe/Qwen3-14B-Instruct",
    )

    # Initialize the server
    backend = LocalBackend()

    await model.register(backend)

    for i in range(await model.get_step(), 20):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, Scenario2048(step=i)) for _ in range(18)
                )
                for _ in range(1)
            ),
            pbar_desc="gather",
            max_exceptions=18,
        )
        await model.delete_checkpoints("train/reward")
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=1e-5),
        )


if __name__ == "__main__":
    asyncio.run(main())
