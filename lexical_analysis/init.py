import copy


""" Ce module regroupe les fonctions permettant d'initialiser les tables de transitions des differents automates """


def initNum(AutNum):
	
	""" Initialise l'automate reconnaissant les chiffres """
	
	AutNum.F = [1,2]
	for i in range(10):
		AutNum.delta[0][ord(str(i))] = 1
		AutNum.delta[1][ord(str(i))] = 1
		AutNum.delta[2][ord(str(i))] = 2
		AutNum.delta[3][ord(str(i))] = 2
	AutNum.delta[0][ord('.')] = 3
	AutNum.delta[1][ord('.')] = 2
	
	
	#~ AutNum.delta[0][ord('-')] = 3
	#~ AutNum.delta[1][ord('-')] = -1
	#~ AutNum.delta[2][ord('-')] = -1
	#~ AutNum.delta[3][ord('-')] = -1
	#~ 
	#~ AutNum.delta[0][ord('+')] = 3
	#~ AutNum.delta[1][ord('+')] = -1
	#~ AutNum.delta[2][ord('+')] = -1
	#~ AutNum.delta[3][ord('+')] = -1

def initCar(AutCar):
	
	""" Initialise l'automate reconnaissant les chaines de caracteres """
	
	AutCar.F = [2]
	
	#Partie du code qui va reconnaitre les chaines a double guillemets  "sdfsdfsdf"
	AutCar.delta[0][ord("\"")] = 1
	
	for i in range(256):
		AutCar.delta[1][i] = 1
	AutCar.delta[1][ord("\"")] = 2
	AutCar.delta[1][ord("\\")] = 3
	
	for i in range(256):
		AutCar.delta[3][i] = 1
	
	
	
	#Partie du code qui va reconnaitre les chaines a simple guillemets  'sdfsdf'
	AutCar.delta[0][ord("'")] = 4
	
	for i in range(256):
		AutCar.delta[4][i] = 4
	AutCar.delta[4][ord("'")] = 2
	AutCar.delta[4][ord("\\")] = 5
	
	for i in range(256):
		AutCar.delta[5][i] = 4

def initTab(Auto, listeMot):
	
	""" Permet d'initialiser un automate de telle sorte qu'il ne reconnaisse que les mots dans le tableau donne en parametre """
	
	nbEtat = 0 #Nombre d'etats total
	for mot in listeMot:
		i=0
		etatActuel = 0
		while i < len(mot): #On parcoure caractere apres caractere
			etat = Auto.delta[etatActuel][ord(mot[i])]  #Etat de l'automate apres lecture du caractere.
			if etat != -1:  #Si la transition a deja etait defini pour ce caractere, on va a l'etat deja existant
				etatActuel = etat
			else:         #Sinon on fais une transition vers un etat encore inutilise, et on repars de ce nouvel etat.
				nbEtat += 1
				Auto.delta[etatActuel][ord(mot[i])] = nbEtat
				etatActuel = nbEtat
			i+=1
		Auto.F += [etatActuel]
		
	
def initId(AutId, AutCl):
	
	"""Reconnait les identificateurs """
	
	AutId.delta = copy.deepcopy(AutCl.delta) #On recopie la table de transition de AutCl dans AutId.
	AutId.delta+= [[-1]*256]  #On lui ajoute un etat tout a la fin qui sera final.
	i=0
	
	for j in range(1,AutId.nbEtat):  #Les etats non acceptants de AutCl sont acceptants dans AutId et inversement. (Sauf l'initial)
		if j not in AutCl.F:
			AutId.F += [j]
	
	
		
	while i < AutId.nbEtat:
		for j in range(ord('a'), ord('z')+1):
			if AutId.delta[i][j] == -1: #Si cette transition, n'a pas ete definie, on la definie, car on sait que le mot qu'on lis ne pourra pas etre un mot-clef
				AutId.delta[i][j] = AutId.nbEtat-1
		
		for j in range(ord('A'), ord('Z')+1):
			if AutId.delta[i][j] == -1: 
				AutId.delta[i][j] = AutId.nbEtat-1
		
		for j in range(ord('0'), ord('9')+1):
			if AutId.delta[i][j] == -1: 
				AutId.delta[i][j] = AutId.nbEtat-1
		
		if AutId.delta[i][ord('_')] == -1: 
				AutId.delta[i][ord('_')] = AutId.nbEtat-1
		i+=1 
			
	for j in range(ord('0'), ord('9')+1):  #Pour l'etat initial, on redefini les transtions pour les chiffres (une variable ne peux pas commencer pas par un chiffre)
			AutId.delta[0][j] = -1
	
	
