# Quantum-classical Tic Tac Toe game with GUI
# using tkinter

# importing all necessary libraries
import random
import tkinter
from tkinter import *
from functools import partial
from tkinter import messagebox
from copy import deepcopy

# sign variable to decide the turn of which player
sign = 0

# Creates an empty board
global board
board = [[" " for x in range(3)] for y in range(3)]

# Check l(O/X) won the match or not
# according to the rules of the game
def winner(b, l):
	return ((b[0][0] == l and b[0][1] == l and b[0][2] == l) or
			(b[1][0] == l and b[1][1] == l and b[1][2] == l) or
			(b[2][0] == l and b[2][1] == l and b[2][2] == l) or
			(b[0][0] == l and b[1][0] == l and b[2][0] == l) or
			(b[0][1] == l and b[1][1] == l and b[2][1] == l) or
			(b[0][2] == l and b[1][2] == l and b[2][2] == l) or
			(b[0][0] == l and b[1][1] == l and b[2][2] == l) or
			(b[0][2] == l and b[1][1] == l and b[2][0] == l))

# Configure text on button while playing with another player
def get_text(i, j, gb, l1, l2):
	global sign
	if board[i][j] == ' ':
		if sign % 2 == 0:
			l1.config(state=DISABLED)
			l2.config(state=ACTIVE)
			board[i][j] = "X"
		else:
			l2.config(state=DISABLED)
			l1.config(state=ACTIVE)
			board[i][j] = "O"
		sign += 1
		button[i][j].config(text=board[i][j])
	
	if winner(board, "X"):
		gb.destroy()
		box = messagebox.showinfo("Winner", "Player 1 won the match")
	elif winner(board, "O"):
		gb.destroy()
		box = messagebox.showinfo("Winner", "Player 2 won the match")
	elif(isfull()):
		gb.destroy()
		box = messagebox.showinfo("Tie Game", "Tie Game")

# Check if the player can push the button or not
def isfree(i, j):
	return board[i][j] == " "

# Check the board is full or not
def isfull():
	flag = True
	for i in board:
		if(i.count(' ') > 0):
			flag = False
	return flag

# Create the GUI of game board for play along with another player
def gameboard_pl(game_board, l1, l2):
	global button
	button = []
	for i in range(3):
		m = 3+i
		button.append(i)
		button[i] = []
		for j in range(3):
			n = j
			button[i].append(j)
			get_t = partial(get_text, i, j, game_board, l1, l2)
			button[i][j] = Button(
				game_board, bd=5, command=get_t, height=4, width=8)
			button[i][j].grid(row=m, column=n)
	game_board.mainloop()



	

# Initialize the game board to play with another player
#Quantum mode button 

def withplayer(game_board):
	game_board.destroy()
	game_board = Tk()
	game_board.title("Tic Tac Toe")
	l1 = Button(game_board, text = "Player 1 : X", width = 10)
	
	l1.grid(row = 1, column = 1)
	l2 = Button(game_board, text = "Player 2 : O",
				width = 10, state = DISABLED)
	
	l2.grid(row = 2, column = 1)

	def change_mode(): 
		if(switch['text']=='Quantum mode is off'):
			switch['text'] = 'Quantum mode is on'
		else:
			switch['text'] = 'Quantum mode is off'
	
	switch = Button(game_board, text='Quantum mode is off', command = change_mode)
	switch.grid(row = 0, column = 1)
	gameboard_pl(game_board, l1, l2)

# main function
def play():
	menu = Tk()
	menu.geometry("400x400")
	menu.title("Quantum Tic Tac Toe")
	
	wpl = partial(withplayer, menu)
	
	head = Button(menu, text = "Welcome to Quantum tic-tac-toe",
				activeforeground = 'white',
				activebackground = "blue", bg = "green",
				fg = "black", width = 500, font = 'summer', bd = 5)
	
	
	
	B2 = Button(menu, text = "Multi Player", command = wpl, activeforeground = 'white',
				activebackground = "blue", bg = "green", fg = "white",
				width = 500, font = 'summer', bd = 5)
	
	B3 = Button(menu, text = "Exit", command = menu.quit, activeforeground = 'white',
				activebackground = "blue", bg = "green", fg = "white",
				width = 500, font = 'summer', bd = 5)

	head.pack(side = 'top')
	B2.pack(side = 'top')
	B3.pack(side = 'top')
	
	menu.mainloop()

# Call main function
if __name__ == '__main__':
	play()
