"""
Number 5 answer

"""

import string, math, random

class Card(object):
    """
    Card class
    Ranks : card number (Example 1,2, .... J = 10 Q = 12 A = 14)
    Suits : Spade, Diamond, Heart, Clover
    """

    RANKS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
    SUITS = ('S', 'D', 'H', 'C')

    def __init__(self, rank, suit):
        """
        Create the card class.

        """
        self.rank = rank
        self.suit = suit

    def __str__(self):
        """The function to check if the file exists

        Return:
             A type of Integer made from between 1 and  A, K, Q, J
        """

        if self.rank == 14:
            rank = 'A'
        elif self.rank == 13:
            rank = 'K'
        elif self.rank == 12:
            rank = 'Q'
        elif self.rank == 11:
            rank = 'J'
        elif self.rank == 10:
            rank = 'T'
        else:
            rank = self.rank
        return str(rank) + self.suit


class Deck(object):
    """
        Deck class
        The class has the card class

    """

    def __init__(self):
        """
        Create the Deck class from card classes

        The list in all kinds of suits of cards and number of card

        """
        self.deck = []
        for suit in Card.SUITS:
            for rank in Card.RANKS:
                card = Card(rank, suit)
                self.deck.append(card)

    def shuffle(self):
        """
            Shuffle deck list using ramdom function
        """
        random.shuffle(self.deck)

    def __len__(self):
        """ Define length of list

        Return:
             A type integer length of the deck list
        """
        return len(self.deck)

    def deal(self):
        """ Serving a card from deck list.

        Return:
             A class of card from dec list.
        """
        if len(self) == 0:
            return None
        else:
            return (self.deck.pop(0)).__str__()


class Poker(object):
    """
        Poker class
        Magage Poker game

    """
    def __init__(self, numHands):
        """
        Create Poker game management class

        Make deck class and shuffle
        Initial hands list
        Initial total score list

        Setting number of cards is 5

        Making hand list which have 5 card

         Args:
              numHands : Number of players between 2 and 6

        """
        self.deck = Deck()
        self.deck.shuffle()
        self.hands = []
        self.totalscorelist = []
        numCards_in_Hand = 5
        self.poker_evaluation = ''
        self.rankvalues = dict((r, i)
                          for i, r in enumerate('..23456789TJQKA'))
        for i in range(numHands):
            hand = []
            for j in range(numCards_in_Hand):
                hand.append(self.deck.deal())
            self.hands.append(hand)

    def play(self):
        """ Print hand which is sorted

        """
        for i in range(len(self.hands)):
            sortedHand = sorted(self.hands[i], reverse=True)
            hand = ''
            for card in sortedHand:
                hand = hand + str(card) + ' '
            print('Hand ' + str(i + 1) + ': ' + hand)

    def evaluate(self, hands):
        """ Evaluate hands

        """
        rank_of_hand = ['high card', 'kind','2 pair', '3 of a kind', 'straight', 'flush', 'full house', '4 of a kind', 'straight flush']
        def hand_rank(hand):
            """ Make poker ranking

            """
            suits = [s for r, s in hand]
            ranks = sorted([self.rankvalues[r] for r, s in hand])
            ranks.reverse()
            flush = len(set(suits)) == 1
            straight = (max(ranks) - min(ranks)) == 4 and len(set(ranks)) == 5

            def kind(n, butnot=None):
                for r in ranks:
                    if ranks.count(r) == n and r != butnot: return r
                return None

            if straight and flush: return (9, ranks), 'straight flush'
            if kind(4): return (8, kind(4), kind(1)), '4 of a kind'
            if kind(3) and kind(2): return (7, kind(3), kind(2))
            if flush: return (6, ranks)
            if straight: return (5, ranks)
            if kind(3): return (4, kind(3), ranks)
            if kind(2) and kind(2, kind(2)): return (3, kind(2), kind(2, kind(2)), ranks)
            if kind(2): return (2, kind(2), ranks)
            return (1, ranks)
        for i, hand in enumerate(hands):
            print("Hand {0} -->  \"{1}\"\t\t info {2}".format(i+1,rank_of_hand[hand_rank(hand)[0]-1],hand_rank(hand)))
        return "## Reuslt \nHand {0} win".format(hands.index( max(hands, key=hand_rank))+1)


def main():
    """ Main

    Input number of players between 2 and 6 because of 1 deck.
    Show hand which is playes got
    Evaluating hands
    Show which hands win

    """
    numHands = eval(input('Enter number of player: '))
    while (numHands < 2 or numHands > 10):
         numHands = eval(input('Enter number of hands to play[2~6]: '))
    game = Poker(numHands)
    print("## Game start")
    game.play()
    print("## Caculating hands")
    print(game.evaluate(game.hands))

if __name__ == "__main__":
    """ Main
    """
    main()

