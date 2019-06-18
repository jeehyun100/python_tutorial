import random


class Card(object):
    def __init__(self, suit, val):
        self.suit = suit
        self.value = val

    def __unicode__(self):
        return self.show()

    def __str__(self):
        return self.show()

    def __repr__(self):
        return self.show()

    def show(self):
        if self.value == 1:
            val = "Ace"
        elif self.value == 11:
            val = "Jack"
        elif self.value == 12:
            val = "Queen"
        elif self.value == 13:
            val = "King"
        else:
            val = self.value
        return "{} of {}".format(val, self.suit)


class Deck(object):
    def __init__(self):
        self.cards = []
        self.build()
    def show(self):
        for card in self.cards:
            print(card.show())
    def build(self):
        self.cards = []
        for suit in ['Hearts', 'Clubs', 'Diamonds', 'Spades']:
            for val in range(1, 14):
                self.cards.append(Card(suit, val))

    def shuffle(self, num=1):
           length = len(self.cards)
           for _ in range(num):
               random.shuffle(self.cards)

    def deal(self):
        return self.cards.pop()

class Player(object):
    def __init__(self, name):
        self.name = name
        self.hand = []
    def sayHello(self):
        print("Hi! My name is {}".format(self.name))
        return self
    def draw(self, deck, num=1):
        for _ in range(num):
            card = deck.deal()
            if card:
                self.hand.append(card)
            else:
                return False
        return True
    def showHand(self):
        print("{}'s hand: {}".format(self.name, self.hand))

        return self

    def discard(self):
        return self.hand.pop()

    def rank(self):
        """ Evaluate hands

        """
        rank_of_hand = ['high card', 'One Pair','Two pair', 'Three of a kind', 'Straight', 'Flush', 'Full house', 'Four of a kind', 'Straight flush']
        def hand_rank():
            """ Make poker ranking

            """
            ordered_values = ['.','.','2','3','4','5','6','7','8','9','10','11', '12','13', '1']
            self.rankvalues2 = dict((r, i)
                                   for i, r in enumerate('..23456789TJQKA'))
            self.rankvalues = dict((r, i)
                                   for i, r in enumerate(ordered_values))
            suits = [item.suit for item in self.hand]
            #ranks = sorted([self.rankvalues[r] for r, s in self.hand])
            ranks = sorted([self.rankvalues[str(item.value)] for item in self.hand])
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
        #for i, hand in enumerate(self.hands):
        print("Hand {0} -->  \"{1}\"\t\t info {2}".format(1,rank_of_hand[hand_rank()[0]-1],hand_rank()))
        #return "## Reuslt \nHand {0} win".format(hands.index( max(hands, key=hand_rank))+1)

    def manual_rank(self, manual_rank):
        """ Evaluate hands

        """
        rank_of_hand = ['high card', 'One Pair','Two pair', 'Three of a kind', 'Straight', 'Flush', 'Full house', 'Four of a kind', 'Straight flush']
        def manual_hand_rank():
            """ Make poker ranking

            """
            ordered_values = ['.','.','2','3','4','5','6','7','8','9','10','11', '12','13', '1']
            self.rankvalues2 = dict((r, i)
                                   for i, r in enumerate('..23456789TJQKA'))
            self.rankvalues = dict((r, i)
                                   for i, r in enumerate(ordered_values))
            suits = [item.suit for item in manual_rank]
            #ranks = sorted([self.rankvalues[r] for r, s in self.hand])
            ranks = sorted([self.rankvalues[str(item.value)] for item in manual_rank])
            ranks.reverse()
            flush = len(set(suits)) == 1
            straight = (max(ranks) - min(ranks)) == 4 and len(set(ranks)) == 5

            def kind(n, butnot=None):
                for r in ranks:
                    if ranks.count(r) == n and r != butnot: return r
                return None
            if straight and flush and len(set(suits).difference(set(['Spades']))) == 0: return (10, ranks), 'Royal straight flush'
            if straight and flush: return (9, ranks), 'Straight flush'
            if kind(4): return (8, kind(4), kind(1)), 'Four of a kind'
            if kind(3) and kind(2): return (7, kind(3), kind(2)),'Full house'
            if flush: return (6, ranks),'Flush'
            if straight: return (5, ranks) ,'Straight'
            if kind(3): return (4, kind(3), ranks), 'Three of a kind'
            if kind(2) and kind(2, kind(2)): return (3, kind(2), kind(2, kind(2)), ranks) ,'Two pair'
            if kind(2): return (2, kind(2), ranks),  'One Pair'
            return (1, ranks), 'high card',
        #for i, hand in enumerate(self.hands):
        print("Hand {0} -->  \{1}".format(1,manual_hand_rank()))
        #return "## Reuslt \nHand {0} win".format(hands.index( max(hands, key=hand_rank))+1)
myDeck = Deck()
myDeck.shuffle()
bob = Player("Bob")
bob.sayHello()
bob.draw(myDeck, 5)
bob.showHand()
bob.rank()

#manual
cards = []
cards.append(Card('Spades', 1))
cards.append(Card('Spades', 13))
cards.append(Card('Spades', 12))
cards.append(Card('Spades', 11))
cards.append(Card('Spades', 10))
bob.manual_rank(cards)


cards = []
cards.append(Card('Spades', 13))
cards.append(Card('Hearts', 13))
cards.append(Card('Clubs', 13))
cards.append(Card('Diamonds', 13))
cards.append(Card('Spades', 12))
bob.manual_rank(cards)
