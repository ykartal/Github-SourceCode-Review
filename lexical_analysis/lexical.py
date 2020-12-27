import sys

from constantes import *
import h5py
import pandas as pd
import numpy as np
import logging



def Appliquer(auto, char):
    """ Applique une chaine a un automate et renvoie le dernier etat acceptant (ou -1 s'il n'y en a pas) """
    etat = 0
    i = 0
    final = -1
    while i < len(char) and etat != -1:
        # ~ print char[i], etat, auto.delta[etat][ord(char[i])]
        #print("index: ", i)
        #print("Etat: ", etat)
        #print("char: ", char[i])
        #print("ord(char[i])", ord(char[i]))
        try:
            etat = auto.delta[etat][ord(char[i])]
            if etat in auto.F:
                final = i
            i += 1
        except:
            final = i
            break

    return final


def appliquerAutomates(chaine):
    """ Permet d'appliquer la chaine sur tous les automates """

    listeEtatsFinaux = [Appliquer(AutNum, chaine), Appliquer(AutCar, chaine), Appliquer(AutSymb, chaine),
                        Appliquer(AutCl, chaine), Appliquer(AutId, chaine)]

    # On cherche maintenant le max dans cette liste
    max = listeEtatsFinaux[0]
    indiceDuMax = 0
    for i in range(1, len(listeEtatsFinaux)):
        if listeEtatsFinaux[i] > max:
            max = listeEtatsFinaux[i]
            indiceDuMax = i
    return listeEtatsFinaux[indiceDuMax], typeUnLex[
        indiceDuMax]  # On retourne la plus graned valeur que retourne Appliquer pour les automates, et le type de l'automates qui a cette plus grande valeur.



def AnLex(listProg):
    global i
    retraitCourant = 0
    lexUnits = []
    i = 0
    j = 0
    while j < len(listProg):  # On parcourt le programme ligne apres ligne
        ligne = listProg[j]
        i=0
        newRetrait = 0

        # On compte le retrait
        while i < len(ligne) and ligne[i] == '\t':
            if ligne[i] == '\t':
                newRetrait += 1
            i += 1

        if i < len(ligne) and not detecteComments(listProg, j,i):  # Si le retrait n'est pas suivi d'un commentaire ou d'un retour a la ligne
            if newRetrait - retraitCourant <= 1:  # Si on a moins de deux debuts de procedures, c'est bon.
                listeLex = debut_fin_procedure(newRetrait - retraitCourant, lexUnits)
                retraitCourant = newRetrait

                # Ici, on s'est occupe du retrait, donc on peut passer a la suite de la ligne.
                while i < len(ligne):
                    if not detecteComments(listProg, j, i) and not detecteSeparateur(listProg, j, i):
                        (positionTmp, type) = appliquerAutomates(ligne[i:])
                        if positionTmp == -1:  # Si aucun automate ne reconnait la chaine :
                            return j, i, MotInvalideError, lexUnits
                        lex = UnLex(type, ligne[i:i + positionTmp + 1])
                        # print("On trouve ("+lex.label+", "+lex.type+") dans :", ligne[i:])
                        i += positionTmp

                        lexUnits += [lex]


                    else:
                        if ligne[i] == '#':
                            break

                        elif not detecteSeparateur(listProg, j, i):  # Si on a detecte une chaine multilignes :
                            if cherch3Quotes(listProg, j, i + 3) == False:  # Si cette chaine n'est pas fermee.
                                return j, i, ChaineMultiligneError, lexUnits
                            else:  # On va ajouter la chaine multilignes a la liste des unites lexicales.
                                (jprime, iprime) = cherch3Quotes(listProg, j, i + 3)
                                chaine = "\""
                                i += 3
                                while j < jprime:
                                    chaine += listProg[j][i:] + "\\n"
                                    i = 0
                                    j += 1
                                chaine += listProg[j][i:iprime] + "\""
                                lexUnits += [UnLex('Car', chaine)]
                                print("On trouve (" + chaine + ", Car). (Chaine multilignes)")
                                (j, i) = (jprime, iprime)
                                ligne = listProg[j]
                                i += 2

                    i += 1

            else:  # Si on detecte deux debuts de procedures consecutives, on s'arrete car c'est impossible.
                return j, i, IndentationError, lexUnits

        else:
            if i < len(ligne) and ligne[
                i] != '#':  # Si on rencontre un commentaire multiligne en debut de ligne ou apres un retrait.
                if cherch3Quotes(listProg, j,
                                 i + 3) == False:  # Si on ouvre un commentaire multiligne sans le fermer plus tard dans le code :
                    return j, i, ChaineMultiligneError, lexUnits
                else:  # Sinon on va a l'endroit ou ce commentaire est ferme.
                    (j, i) = cherch3Quotes(listProg, j, i + 3)
        lexUnits += [UnLex('Symb', '\n')]
        j += 1

    listeLex = debut_fin_procedure(-retraitCourant, lexUnits)
    return j, i, True, lexUnits


def debut_fin_procedure(difference, listeLex):
    """ Recoit la difference entre le retraitCourant et le retrait lu, et ecrit les debuts ou fin de procedure en consequences """

    if difference == 1:
        lex = UnLex('Pr', '${')
        listeLex += [lex]
    elif difference < 0:
        i = difference
        while i < 0:
            lex = UnLex('Pr', '}$')
            listeLex += [lex]
            i += 1

    return listeLex


def detecteComments(liste, j, i):
    """ Retourne True si le caractere courant est un commentaire """

    return liste[j][i] == '#' or (
                i < len(liste[j]) - 2 and liste[j][i] == "\"" and liste[j][i + 1] == "\"" and liste[j][i + 2] == "\"")


def detecteSeparateur(liste, j, i):
    """ Retourne True si le caractere courant est une tabulation ou un espace """
    return liste[j][i] in [' ', '\t']


def cherch3Quotes(liste, j, i):
    """ Prend une position dans le code python a analyser et renvoie la position du prochain caractere : \"""  """

    nbGuillemets = 0  # Sert a compter les guillemets doubles consecutifs.

    while j < len(liste):
        while i < len(liste[j]) - 2:
            if liste[j][i] == "\"" and liste[j][i + 1] == "\"" and liste[j][i + 2] == "\"":
                return (j, i)  # On retourne la position du premier guillemet du groupe de 3
            i += 1
        i = 0
        j += 1
    return False


def diviseLignes(prog):
    """ Divise une chaine de caracteres par ligne, retourne un tableau ou chaque element est une ligne de prog """

    liste = []
    ligne = ""
    i = 0
    while i < len(prog):
        if prog[i] == '\n':  # Si on rencontre un saut de ligne.
            # if ligne != "": # Et si cette ligne precedant le saut de lignes n'etait pas vide, on l'ajoute.
            liste += [ligne]
            ligne = ""
        else:  # Si on atteit pas un saut de ligne, on continue a lire
            ligne += prog[i]
        i += 1
    if ligne != "":
        liste += [ligne]
    return liste


def to_lex(inF):
    output = ""
    # On ouvre les fichiers en arguments.
    try:
        outF = open('out', "w+")
    except IOError:
        print(
            "\nVerifiez que vous avez bien les droits d'ecritures dans le dossier et que le fichier en entre existe bien.\n")

    # print("Initialisation des automates en cours ...")
    # temps = time.clock()
    # print("Automates pret en " + str(time.clock() - temps) + " secondes !")

    # t = time.clock()
    listProg = diviseLignes(inF)  # Liste contenant le programme divise par lignes.
    numLigne, numColonnes, etat, lexique = AnLex(listProg)
    # print("\nAnalyse finie en " + str(time.clock() - t) + " secondes.")

    # On regarde si l'execution de l'analyseur c'est bien passee.
    if etat == True:
        print("Tout c'est bien passe.")
        for i in lexique:
            if i.label != '\n':
                output = output + "(" + i.label + ", " + i.type + ") "
                outF.write("(" + i.label + ", " + i.type + ") ")
            else:
                outF.write(
                    "(\\n, " + i.type + ") ")  # Pour afficher \n dans un fichier, il faut lui demander d'ecrire \\n  sinon ca ne marche pas.
                output = output + "(\\n, " + i.type + ") "
            if i.label in ['\n', '${',
                           '}$']:  # Si on detecte un retour a la ligne ou un debut/fin de procedure, on fais un retour a la ligne.
                outF.write("\n")
                output = output + "\n"

    else:
        print("")
        # print("\n\nErreur a la ligne " + str(numLigne + 1) + ", caractere", str(numColonnes + 1) + ".")
        # print(Dictionnaire_Erreurs[etat], "\n")
        # print(listProg[numLigne])
        # print(" " * numColonnes + "^")
    return output


