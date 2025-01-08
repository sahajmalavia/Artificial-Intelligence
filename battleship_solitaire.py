from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from typing import List, Set, Tuple, Optional

Position = Tuple[int, int]
Battleship_Value = Tuple[Position, int]


class ConstraintSatisfactionProblem:
    def __init__(self, filename: str):
        self.read_input_file(filename)
        self.preprocess()

    def is_surrounded_by_water(self, col: int, row: int, orientation: Tuple[int, int] = (1, 0)) -> bool:
        cell = self[col, row]
        directions = self._get_surrounding_directions(cell, orientation)

        for dx, dy in directions:
            nx, ny = col + dx, row + dy
            if self.is_valid_index(nx, ny) and self[nx, ny] != ".":
                return False
        return True

    # Helper
    def _get_surrounding_directions(self, cell: str, orientation: Tuple[int, int]) -> List[Tuple[int, int]]:
        direction_map = {
            "S": [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)],
            "^": [(1, 0), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)],
            "v": [(1, 0), (-1, 0), (0, 1), (1, 1), (-1, 1), (-1, -1), (1, -1)],
            "<": [(-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)],
            ">": [(1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)],
            "M": [
                (1, 1), (-1, 1), (-1, -1), (1, -1),
                (orientation[1], orientation[0]), (-orientation[1], orientation[0])
            ],
        }
        return direction_map.get(cell, [])

    def completed(self) -> bool:
        return all(all(v.is_assigned() for v in vs) for vs in self.variables)

    def assign_value(self, var: Variable, value: Battleship_Value) -> bool:
        var.assign(value)
        return self.attempt_placing(var)

    def unassign(self, var: Variable):
        self.remove_ship(var)
        var.unassign()

    def select_variable(self) -> Variable:
        for vars in self.variables[::-1]:
            for var in vars:
                if not var.is_assigned():
                    return var

    def valid_ships_check(self) -> bool:
        visited_segments = set()
        ship_counts = [0] * len(self.ship_constraints)
        row = 0

        while row < self.height:
            col = 0
            while col < self.width:
                current_cell = self[col, row]
                if current_cell == "0":
                    return False
                if current_cell == ".":
                    col += 1
                    continue
                if current_cell == "M" and (col, row) not in visited_segments:
                    return False
                if current_cell == "S":
                    if not self.is_surrounded_by_water(col, row):
                        return False
                    ship_counts[0] += 1
                    if ship_counts[0] > self.ship_constraints[0]:
                        return False
                elif current_cell in {"<", "^"}:
                    if not self.is_surrounded_by_water(col, row):
                        return False
                    ship_length, next_index = 1, (col + 1 if current_cell == "<" else row + 1)
                    while next_index < (self.width if current_cell == "<" else self.height) and \
                            self[(next_index, row) if current_cell == "<" else (col, next_index)] == "M":
                        if not self.is_surrounded_by_water(next_index, row,
                                                           (1, 0) if current_cell == "<" else (0, 1)):
                            return False
                        visited_segments.add((next_index, row) if current_cell == "<" else (col, next_index))
                        next_index += 1
                        ship_length += 1
                    end_part = (self[next_index, row] if current_cell == "<" else self[col, next_index]) \
                        if next_index < (self.width if current_cell == "<" else self.height) else None
                    if (end_part != (">" if current_cell == "<" else "v")) or ship_length >= len(
                            self.ship_constraints):
                        return False
                    ship_counts[ship_length] += 1
                    if ship_counts[ship_length] > self.ship_constraints[ship_length]:
                        return False
                col += 1
            row += 1

        return ship_counts == self.ship_constraints

    def validate_constraints(self) -> bool:

        row_counts = [0] * self.height
        col_counts = [0] * self.width
        row = 0

        while row < self.height:
            col = 0
            while col < self.width:
                cell = self[col, row]
                if cell == "0":
                    return False
                if cell == ".":
                    col += 1
                    continue
                row_counts[row] += 1
                if row_counts[row] > self.row_constraints[row]:
                    return False
                col_counts[col] += 1
                if col_counts[col] > self.column_constraints[col]:
                    return False
                col += 1
            row += 1

        return row_counts == self.row_constraints and col_counts == self.column_constraints

    def attempt_placing(self, ship: Variable) -> bool:
        row_counts, col_counts = self.count_parts_by_RowAndCol()
        start_col, start_row = ship.position
        ship_length = ship.size
        start_cell = self[start_col, start_row]

        if ship.alignment == 1:
            if start_col + ship_length > self.width:
                return False

            has_empty_cell = False
            row_delta = 0
            col_deltas = [0] * self.width

            # Check first cell
            if start_cell == "0":
                row_delta += 1
                col_deltas[start_col] += 1
                has_empty_cell = True
            elif start_cell != "<":
                return False

            # Check middle cells
            current_col = start_col + 1
            while current_col < start_col + ship_length - 1:
                cell = self[current_col, start_row]
                if cell == "0":
                    row_delta += 1
                    col_deltas[current_col] += 1
                    has_empty_cell = True
                elif cell != "M":
                    return False
                current_col += 1

            # Check last cell
            last_cell = self[start_col + ship_length - 1, start_row]
            if last_cell == ">":
                if not has_empty_cell:
                    return False
            elif last_cell != "0":
                return False
            else:
                row_delta += 1
                col_deltas[start_col + ship_length - 1] += 1

            # Validate row and column constraints
            if row_counts[start_row] + row_delta > self.row_constraints[start_row]:
                return False
            for col in range(self.width):
                if col_counts[col] + col_deltas[col] > self.column_constraints[col]:
                    return False

            # Place the ship
            self.put_ship_on_grid(ship)
            return True

        elif ship.alignment == 2:
            if start_row + ship_length > self.height:
                return False

            has_empty_cell = False
            row_deltas = [0] * self.height
            col_delta = 0

            # Check first cell
            if start_cell == "0":
                row_deltas[start_row] += 1
                col_delta += 1
                has_empty_cell = True
            elif start_cell != "^":
                return False

            # Check middle cells
            current_row = start_row + 1
            while current_row < start_row + ship_length - 1:
                cell = self[start_col, current_row]
                if cell == "0":
                    row_deltas[current_row] += 1
                    col_delta += 1
                    has_empty_cell = True
                elif cell != "M":
                    return False
                current_row += 1

            # Check last cell
            last_cell = self[start_col, start_row + ship_length - 1]
            if last_cell == "v":
                if not has_empty_cell:
                    return False
            elif last_cell != "0":
                return False
            else:
                row_deltas[start_row + ship_length - 1] += 1
                col_delta += 1

            # Validate column and row constraints
            if col_counts[start_col] + col_delta > self.column_constraints[start_col]:
                return False
            for row in range(self.height):
                if row_counts[row] + row_deltas[row] > self.row_constraints[row]:
                    return False

            # Place the ship
            self.put_ship_on_grid(ship)
            return True

        else:
            # Submarine case
            if row_counts[start_row] + 1 > self.row_constraints[start_row]:
                return False
            if col_counts[start_col] + 1 > self.column_constraints[start_col]:
                return False
            if self[start_col, start_row] == "0":
                self.put_ship_on_grid(ship)
                return True

            return False

    def count_parts_by_RowAndCol(self) -> Tuple[List[int], List[int]]:
        row_counts = [0] * self.height
        col_counts = [0] * self.width
        row = 0

        while row < self.height:
            col = 0
            while col < self.width:
                cell = self[col, row]
                if cell != "0" and cell != ".":
                    row_counts[row] += 1
                    col_counts[col] += 1
                col += 1
            row += 1

        return row_counts, col_counts

    def put_ship_on_grid(self, ship: Variable): #chagned
        start_col, start_row = ship.position
        length = ship.size

        if ship.alignment == 1:
            self[start_col, start_row] = "<"
            for col in range(start_col + 1, start_col + length - 1):
                self[col, start_row] = "M"
            self[start_col + length - 1, start_row] = ">"

        elif ship.alignment == 2:
            self[start_col, start_row] = "^"
            for row in range(start_row + 1, start_row + length - 1):
                self[start_col, row] = "M"
            self[start_col, start_row + length - 1] = "v"

        else:  # SUBMARINE
            self[start_col, start_row] = "S"

    def remove_ship(self, ship: Variable):
        start_col, start_row = ship.position
        length = ship.size

        if ship.alignment == 1:
            for col in range(start_col, start_col + length):
                self[col, start_row] = self.original_grid[start_row][col]

        elif ship.alignment == 2:
            for row in range(start_row, start_row + length):
                self[start_col, row] = self.original_grid[row][start_col]

        else:  # SUBMARINE
            self[start_col, start_row] = self.original_grid[start_row][start_col]



    def fill_with_water(self):

        self.backup_grid = [row[:] for row in self.grid]  # Create a deep copy of the grid

        for row in range(self.height):
            for col in range(self.width):
                if self[col, row] == "0":
                    self[col, row] = "."

    def water_fill_directions(self, part: str):
        direction_map = {
            "S": [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)],
            "<": [(-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)],
            ">": [(1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)],
            "^": [(1, 0), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)],
            "v": [(1, 0), (-1, 0), (0, 1), (1, 1), (-1, 1), (-1, -1), (1, -1)],
            "M": [(1, 1), (1, -1), (-1, 1), (-1, -1)],
        }
        return direction_map.get(part, [])

    def FC(self, ship: Variable):
        start_col, start_row = ship.position
        ship_length = ship.size
        orientation = ship.alignment
        affected_cells = self._get_ship_locations(start_col, start_row, ship_length, orientation)
        restores = set()

        for col, row in affected_cells:
            cell = self[col, row]
            affected_locations = self._get_affected_locations(cell, col, row, affected_cells)
            affected_domain = set(itertools.product(
                affected_locations,
                (0, 1, 2),
            ))

            for vars_group in self.variables:
                for var in vars_group:
                    if not var.is_assigned() and var.reduceDomain(affected_domain, ship, ship.current_value()):
                        restores.add(var)

        return restores

    # Helpers
    def _get_ship_locations(self, start_col, start_row, ship_length, orientation):
        if orientation == 1:
            return [(col, start_row) for col in range(start_col, start_col + ship_length)]
        elif orientation == 2:
            return [(start_col, row) for row in range(start_row, start_row + ship_length)]
        else:  # SUBMARINE
            return [(start_col, start_row)]

    def _get_affected_locations(self, cell, col, row, affected_cells):
        directions = self.water_fill_directions(cell)
        affected_locations = [(col + dx, row + dy) for dx, dy in directions] + affected_cells
        return affected_locations

    def is_valid_index(self, col: int, row: int) -> bool:
        return 0 <= col < self.width and 0 <= row < self.height

    def read_input_file(self, filename: str):
        with open(filename) as file:
            raw = file.readlines()

        # Parse constraints
        self.row_constraints = list(map(int, raw[0].strip()))
        self.column_constraints = list(map(int, raw[1].strip()))
        self.ship_constraints = list(map(int, raw[2].strip()))

        # Parse grid
        self.grid = [list(row.strip()) for row in raw[3:]]
        self.original_grid = tuple(tuple(row.strip()) for row in raw[3:])
        self.height, self.width = len(self.grid), len(self.grid[0])

        # Initialize variables
        self._initialize_variables()

    def __setitem__(self, key: Tuple[int, int], value: str):
        col, row = key
        self.grid[row][col] = value

    def _initialize_variables(self):
        self.variables = []
        all_locations = list(itertools.product(range(self.height), range(self.width)))

        # Submarine variables
        submarine_domain = {(loc, 0) for loc in all_locations}
        self.variables.append([
            Variable((-1, -1), -1, 1, submarine_domain.copy())
            for _ in range(self.ship_constraints[0])
        ])

        # Other ship variables
        domain = set(itertools.product(all_locations, (1, 2)))
        for size, count in enumerate(self.ship_constraints[1:], start=2):
            self.variables.append([
                Variable((-1, -1), -1, size, domain.copy())
                for _ in range(count)
            ])
    def preprocess(self):
        self._fill_surrounding_water()
        self._fill_complete_rows_and_columns()
        self._set_submarine_variables()

    # Helpers
    def _fill_surrounding_water(self):
        for row in range(self.height):
            for col in range(self.width):
                cell = self[col, row]
                if cell in {"0", "."}:
                    continue

                directions_to_fill = self._get_extended_directions(cell)
                for dx, dy in directions_to_fill:
                    nx, ny = col + dx, row + dy
                    if self.is_valid_index(nx, ny) and self[nx, ny] == "0":
                        self[nx, ny] = "."

    def _fill_complete_rows_and_columns(self):
        row_counts, col_counts = self.count_parts_by_RowAndCol()

        for row in range(self.height):
            if row_counts[row] == self.row_constraints[row]:
                for col in range(self.width):
                    if self[col, row] == "0":
                        self[col, row] = "."

        for col in range(self.width):
            if col_counts[col] == self.column_constraints[col]:
                for row in range(self.height):
                    if self[col, row] == "0":
                        self[col, row] = "."

    def _set_submarine_variables(self):
        for row in range(self.height):
            for col in range(self.width):
                cell = self[col, row]
                if cell in {"0", "<", "^"}:
                    continue

                for var_group in self.variables:
                    for variable in var_group:
                        if not variable.is_assigned():
                            variable.reduce_original_domain({
                                ((col, row), 0),
                                ((col, row), 1),
                                ((col, row), 2),
                            })

                if cell == "S":
                    for submarine_var in self.variables[0]:
                        if submarine_var.is_assigned():
                            continue
                        submarine_var.set_the_constant((col, row), 0)
                        break

    def _get_extended_directions(self, cell):
        base_directions = {
            "S": [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)],
            "<": [(-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1), (2, -1), (2, 1)],
            ">": [(1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1), (-2, -1), (-2, 1)],
            "^": [(1, 0), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1), (1, 2), (-1, 2)],
            "v": [(1, 0), (-1, 0), (0, 1), (1, 1), (-1, 1), (-1, -1), (1, -1), (1, -2), (-1, -2)],
            "M": [(1, 1), (1, -1), (-1, 1), (-1, -1)],
        }
        return base_directions.get(cell, [])

    def write_output_to_file(self, filename: str):
        with open(filename, 'w') as file:
            file.writelines(''.join(row) + '\n' for row in self.grid)



    def __getitem__(self, key: Tuple[int, int]) -> str:
        col, row = key
        return self.grid[row][col]

class Variable:

    def __init__(self, coordinate: Position, orientation: int, length: int, domain: Set[Battleship_Value]):
        self.position = coordinate
        self.alignment = orientation
        self.size = length
        self.domain = domain.copy()
        self.original_domain = domain.copy()
        self.pruned = defaultdict(set)
        self.is_fixed = False

    def set_the_constant(self, coordinate: Position = None, orientation: int = None):
        if self.is_fixed:
            return
        self.is_fixed = True
        if coordinate is None:
            coordinate = self.position
        else:
            self.position = coordinate
        if orientation is not None:
            self.alignment = orientation
        self.set_original_domain({(coordinate, orientation)})

    def assign(self, value: Battleship_Value):
        self.position, self.alignment = value

    def unassign(self):
        self.position = (-1, -1)
        self.alignment = -1

    def is_assigned(self) -> bool:
        return self.position != (-1, -1) and \
               self.alignment != -1

    def reduceDomain(self, reduction_factor: Set[Battleship_Value],
                     var: Variable, val: Battleship_Value) -> bool:
        reduction_factor = reduction_factor & self.domain
        if len(reduction_factor) == 0:
            return False
        self.domain -= reduction_factor
        self.pruned[(var, val)] |= reduction_factor
        return True

    def restore_domain(self, var: Variable, val: Battleship_Value):
        self.domain |= self.pruned[(key := (var, val))]
        self.pruned.pop(key)

    def reduce_original_domain(self, reduction_factor: Set[Battleship_Value]):
        self.original_domain -= reduction_factor
        self.restore_original_domain()

    def set_original_domain(self, new_domain: Set[Battleship_Value]):
        self.original_domain = new_domain.copy()
        self.restore_original_domain()

    def restore_original_domain(self):
        self.domain = self.original_domain.copy()

    def current_value(self) -> Battleship_Value:
        return self.position, self.alignment



def BT(csp: ConstraintSatisfactionProblem) -> Optional[ConstraintSatisfactionProblem]:
    if csp.completed():
        return _finalize_solution(csp)

    variable = csp.select_variable()

    for value in variable.domain:
        if not _assign_value_with_check(csp, variable, value):
            continue

        result = BT(csp)
        if result:
            return result

        _undo_assignment(csp, variable, value)

    return None

# Helpers
def _finalize_solution(csp: ConstraintSatisfactionProblem) -> Optional[ConstraintSatisfactionProblem]:
    csp.fill_with_water()
    if csp.validate_constraints() and csp.valid_ships_check():
        return csp
    csp.grid, csp.backup_grid = csp.backup_grid, None
    return None

def _assign_value_with_check(csp: ConstraintSatisfactionProblem, variable, value) -> bool:
    if not csp.assign_value(variable, value):
        variable.unassign()
        return False

    restores = csp.FC(variable)
    variable._restore_points = restores  # Track restores for rollback

    return True

def _undo_assignment(csp: ConstraintSatisfactionProblem, variable, value):
    csp.unassign(variable)
    if hasattr(variable, '_restore_points'):
        for restore_var in variable._restore_points:
            restore_var.restore_domain(variable, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="Path to the input file containing the puzzle.",
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="Path to the output file where the solution will be written.",
    )
    args = parser.parse_args()

    inputFile, outputFile = args.inputfile, args.outputfile
    csp = ConstraintSatisfactionProblem(inputFile)
    solution = BT(csp)

    if solution is None:
        with open(outputFile, "w") as file:
            file.write("No Solution")
        exit(1)

    solution.write_output_to_file(outputFile)
    exit(0)

