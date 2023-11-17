import numpy as np
import tensorflow as tf
import time
import threading
import random
import concurrent.futures
import multiprocessing


# Создаем модель нейронной сети
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(19, 19)),  # Входной слой
    tf.keras.layers.Dense(128, activation='relu'),  # Скрытый слой
    tf.keras.layers.Dense(361, activation='softmax') # Выходной слой
])

# Компилируем модель
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Функция для предсказания вероятностей для каждого возможного хода
def predict_probabilities(board):
    flattened_board = board.flatten().reshape(1, 19, 19)
    return model.predict(flattened_board)[0]

# Функция для выбора лучшего хода с ограничением времени
def select_best_move_with_timeout(board, opponent_last_move, timeout=1.0):
    start_time = time.process_time()

    empty_cells = np.argwhere(board == 0)
    best_move = None
    best_prob = -1

    if len(empty_cells) == 0:
        return None

    for move in empty_cells:
        temp_board = board.copy()
        temp_board[move[0], move[1]] = 1  # Предположим, что ходит первый игрок

        if opponent_last_move is not None:
            temp_board[opponent_last_move[0], opponent_last_move[1]] = -1  # Обновляем последний ход противника

        probabilities = predict_probabilities(temp_board)

        if probabilities[move[0] * 19 + move[1]] > best_prob:
            best_prob = probabilities[move[0] * 19 + move[1]]
            best_move = move

        elapsed_time = time.process_time() - start_time

        if elapsed_time > timeout:
            print("Время на выбор хода превышено.")
            break

    return best_move

# Класс для выполнения хода нейросети в отдельном потоке
class NeuralNetworkThread(threading.Thread):
    def __init__(self, board, opponent_last_move, timeout):
        threading.Thread.__init__(self)
        self.board = board
        self.opponent_last_move = opponent_last_move
        self.timeout = timeout
        self.result = None

    def run(self):
        self.result = select_best_move_with_timeout(self.board, self.opponent_last_move, self.timeout)

# Функция для выполнения хода нейросети с ограничением времени
def neural_network_move_with_timeout(board, opponent_last_move, timeout=1.0):
    with multiprocessing.Pool(1) as pool:
        result = pool.apply_async(select_best_move_with_timeout, (board, opponent_last_move, timeout))
        try:
            neural_network_move = result.get(timeout=timeout)
        except multiprocessing.TimeoutError:
            print("Время на выбор хода нейросети превышено. Выбираем случайный ход.")
            empty_cells = np.argwhere(board == 0)
            neural_network_move = tuple(random.choice(empty_cells))

    return neural_network_move

# Функция для проверки на выигрышную комбинацию
def is_winner(board, player):
    for i in range(15):
        for j in range(15):
            # Проверка по горизонтали
            if np.all(board[i:i+5, j] == player):
                return True
            # Проверка по вертикали
            if np.all(board[i, j:j+5] == player):
                return True
            # Проверка по диагонали (лево-верх -> право-низ)
            if np.all(np.diag(board[i:i+5, j:j+5]) == player):
                return True
            # Проверка по диагонали (лево-низ -> право-верх)
            if np.all(np.diag(np.flipud(board[i:i+5, j:j+5])) == player):
                return True
    return False

# Функция для отображения текущего состояния доски
def print_board(board):
    for row in board:
        print(" ".join(["X" if cell == 1 else "O" if cell == -1 else "." for cell in row]))

# Функция для выполнения хода с ограничением времени
def perform_move(board, player, opponent_last_move, timeout=1.0):
    if player == 1:  # Ход человека
        while True:
            try:
                move = input("Введите ваш ход в формате 'строка столбец' (например, '1 2'): ")
                move = tuple(map(int, move.split()))
                if move in np.argwhere(board == 0):
                    board[move] = player
                    print_board(board)
                    return move
                else:
                    print("Недопустимый ход. Пожалуйста, выберите пустую ячейку.")
            except ValueError:
                print("Ошибка ввода. Пожалуйста, введите два числа через пробел.")

    else:  # Ход нейросети
        thread = NeuralNetworkThread(board, opponent_last_move, timeout)
        thread.start()
        thread.join(timeout)  # Ждем завершения потока или превышения времени

        if thread.is_alive():
            thread._stop()  # Останавливаем поток, если время превышено
            print("Время на выбор хода нейросети превышено. Выбираем случайный ход.")
            empty_cells = np.argwhere(board == 0)
            return tuple(random.choice(empty_cells))
        else:
            neural_network_move = thread.result

        # Если все ячейки заняты, выбираем случайную доступную ячейку
        if board[neural_network_move] != 0:
            empty_cells = np.argwhere(board == 0)
            if len(empty_cells) == 0:
                return None
            neural_network_move = tuple(random.choice(empty_cells))

        board[neural_network_move] = player
        print(f"Игрок {player} выбрал ход {neural_network_move}.")
        print_board(board)
        return neural_network_move

# Пример игры между двумя стратегиями, использующими нейронную сеть для выбора хода
def play_game():
    while True:
        board = np.zeros((19, 19), dtype=int)
        training_data = []  # Добавлен список для сохранения обучающих данных
        while True:
            # Ход первого игрока
            move = perform_move(board, 1, None, 1)
            if move is not None:
                if is_winner(board, 1):
                    print("Первый игрок выиграл!")
                    training_data.append((board.copy(), 1))  # Сохраняем состояние доски и результат игры
                    break

            # Ход второго игрока
            move = perform_move(board, -1, move, 1)
            if move is not None:
                if is_winner(board, -1):
                    print("Второй игрок выиграл!")
                    training_data.append((board.copy(), -1))  # Сохраняем состояние доски и результат игры
                    break

        # Сохраняем обучающие данные в файл
        with open('training_data.txt', 'a') as file:
            for board_state, result in training_data:
                file.write(f"{result} {' '.join(map(str, board_state.flatten()))}\n")

if __name__ == "__main__":
    play_game()
