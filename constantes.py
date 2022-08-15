from init import *
import time

#Ce fichier contient des constantes. 

#Classes
class Automate:
	def __init__(self, n):
		self.delta = []
		self.F = []
		self.nbEtat = n
		for i in range(n):
			ligne = []
			for lettre in range(256):
				ligne+= [-1]
			self.delta+=[ligne]

typeUnLex = ['Num' , 'Car', 'Symb', 'Cl', 'Id', 'Pr']
class UnLex:
	type = ""
	label = ""
	def __init__(self, type, label):
		self.label = label
		if type in typeUnLex:
			self.type = type


#Declaration des erreurs : 
ChaineMultiligneError = -1   #Si on referme pas une chaine multilignes.   """
MotInvalideError = -2        #Si aucun automate n'est acceptant pour un mot donne.

Dictionnaire_Erreurs={  #Associe un message d'erreur a chaque erreur possible.
	IndentationError : "Erreur d'indentation.",  
	ChaineMultiligneError : "Il manque une fin de chaine multilignes (trois guillemets ouverts mais non fermes).",
	MotInvalideError : "Mot non reconnu. Aucun automate n'est acceptant.",
	}

#Declarations des tableaux utilises pour creer les automates AutCl puis AutSymb.
listeMotsClefs = ["and", "as", "assert", "break", "class", "continue", "def", "del", "elif", "else", "except", "exec", "finally", "for", "from", "global", "if", "import", "in", "not", "open", "or", "pass", "print", "raise", "return", "try", "while", "with", "yield", 
"abs", "divmod", "input", "open", "staticmethod", "all", "enumerate", "int", "ord", "str", "any", "eval", "isinstance", "pow", "sum", "basestring", "execfile", "issubclass", "print", "super", "bin", "file", "iter", "property", "tuple", "bool", "filter", "len", "range", "type", "bytearray", "float", "list", "raw_input", "unichr", "callable", "format", "locals", "reduce", "unicode", "chr", "frozenset", "long", "reload", "vars", "classmethod", "getattr", "map", "repr", "xrange", "cmp", "globals", "max", "reversed", "zip", "compile", "hasattr", "memoryview", "round", "__import__", "complex", "hash", "min", "set", "apply", "delattr", "help", "next", "setattr", "buffer", "dict", "hex", "object", "slice", "coerce", "dir", "id", "oct", "sorted", "intern",
"False", "True", "Ellipsis", "NotImplemented", "__debug__", "None",
"BaseException", "SystemExit", "KeyboardInterrupt", "GeneratorExit", "Exception", "StopIteration", "StandardError", "BufferError", "ArithmeticError", "FloatingPointError", "OverflowError", "ZeroDivisionError", "AssertionError", "AttributeError", "EnvironmentError", "IOError", "OSError", "WindowsError(Windows)", "VMSError(VMS)", "EOFError", "ImportError", "LookupError", "IndexError", "KeyError", "MemoryError", "NameError", "UnboundLocalError", "ReferenceError", "RuntimeError", "NotImplementedError", "SyntaxError", "IndentationError", "TabError", "SystemError", "TypeError", "ValueError", "UnicodeError", "UnicodeDecodeError", "UnicodeEncodeError", "UnicodeTranslateError", "Warning", "DeprecationWarning", "PendingDeprecationWarning", "RuntimeWarning", "SyntaxWarning", "UserWarning", "FutureWarning", "ImportWarning", "UnicodeWarning", "BytesWarning"]

listeSymboles = [",", ";", ".", ":", "(", ")", "[", "]", "{", "}", "<", ">", "=", "+", "-", "~", "*", "/", "%", "|", "^", "&", "@", "<<", ">>", "**", "//", "!=","!", "<>", "<=", ">=", "==", "+=", "-=", "~=", "*=", "/=", "%=", "|=", "^=", "&=", "<<=", ">>=", "**=", "//="]




#Declaration et initialisation des automates.
print("Initialisation des automates en cours ...")
temps = time.clock()

AutNum = Automate(4)
initNum(AutNum)
AutCar = Automate(6)
initCar(AutCar)
AutSymb = Automate(46)
initTab(AutSymb, listeSymboles)
AutCl = Automate(1091)
initTab(AutCl, listeMotsClefs)
AutId = Automate(AutCl.nbEtat+1) #Un etat de plus que le precedent automate
initId(AutId, AutCl)
print("Automates pret en "+str(time.clock()-temps)+" secondes !")