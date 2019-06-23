# -*- coding: utf-8 -*-
import random
import unittest


class Card(object):
    """
    Card class
    Card class의 내용을 출력해주는 여러 Magic method가 show()를 참조하도록 구현한다.

    """

    def __init__(self, suit, val):
        """
        생성자
        suit, val을 입력받아 Card class를 생성한다.

        Args:
            suit, String: 카드의 모양, ['Hearts', 'Clubs', 'Diamonds', 'Spades']
            val, Integer : 카드의 숫자, 1~13까지의 숫자

        """
        self.suit = suit
        self.value = val

    def __unicode__(self):
        """
        Card 객체ㅇ를 String 타입으로 출력하기 위한 오버라이딩 method
        show() method를 참조한다.

        """
        return self.show()

    def __str__(self):
        """
        String 형태를 출력하기 위한 오버라이딩 method
        show() method를 참조한다.

        """
        return self.show()

    def __repr__(self):
        """
        Card 객체를 String 타입으로 표현하기 위한 오버라이딩 method
        그러나 show() method를 참조한다

        """
        return self.show()

    def show(self):
        """
        Card 객체의 내용을 보여주는 method

        특정 숫자로 들어온 value를 문자로 변환한다음
        {val} of {suit} 로 문자열을 출력한다.

        return:
            String, {val} of {suit}

        """
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
    """
    Deck class
    Card 객체 및 저장하는 관리하는 기능을 가진 Class

    Card를 생성하여 cards list에 넣어 주는 기능,
    card list를 순회하며 카드 정보를 출력하는 기능을 구현한다.

    """

    def __init__(self):
        """
        생성자
        Desc class를 생성한다.
        Card class를 관리하는 cards list를 만들고,
        build method를 호출하여 card deck을 만든다.

        """
        self.cards = []
        self.build()

    def show(self):
        """
        Deck 객체의 cards list 내용을 보여주는 method
        list를 순회하며 Card class를 출력한다.

        return:
            String, Card class의 show method 예) {val} of {suit}

        """

        for card in self.cards:
            print(card.show())

    def build(self):
        """
         cards list에 Card 객체를 추가하여 cards list를 만든다
         모양은 4개 'Hearts', 'Clubs', 'Diamonds', 'Spades'
         숫자는 14개 1~14로 모든 조합으로 Card 객체를 만들고,
         list에 추가한다.

         """
        self.cards = []
        for suit in ['Hearts', 'Clubs', 'Diamonds', 'Spades']:
            for val in range(1, 14):
                self.cards.append(Card(suit, val))

    def shuffle(self, num=1):
        """
        card list를 섞어준다.

        Args:
            num : Integer, default 1, cards list를 랜덤하게 섞는 횟수

        """
        length = len(self.cards)  # 사용하지 않는 변수 삭제 필요
        for _ in range(num):
            random.shuffle(self.cards)

    def deal(self):
        """
        card list에서 가장 마지막 카드를 반환하고 리스트에서 삭제한다.

        return:
            Card class

        """
        try:
            return self.cards.pop()
        except IndexError:
            print("Error : Deck is empty.")
            raise


class Player(object):
    """
    Player class
    포커 게임을 하는 Player class
    Player의 이름과, Card의 집합인 hand를 가지고 있다.

    Player의 이름을 출력하는 기능,
    특정 수만큼 카드를 Deck으로 부터 받아오는 기능,
    hand list로 부터 카드 정보와 이름을 출력하는 기능,
    hand의 마지막 카드를 가져오는 기능, 포커랭킹을 계산하는 기능을 구현한다.

    """

    def __init__(self, name):
        """
        생성자
        Player class를 생성한다.
        이름과, Card class를 보관한 list를 만든다.

        Args:
            name : string, Player 이름

        """
        self.name = name
        self.hand = []

    def sayHello(self):
        """
        Player의 이름을 출력한다.

        return:
            String, 예) Hi! My name is {name}

        """
        print("Hi! My name is {}".format(self.name))
        return self

    def draw(self, deck, num=1):
        """
        Deck으로 부터 카드를 받는다. 카드가 없으면 False를 반환한다.

        return:
            Boolean, Deck class에서 카드가 없으면 False

        """
        for _ in range(num):
            card = deck.deal()
            if card:
                self.hand.append(card)
            else:
                return False
        return True

    def showHand(self):
        """
        현재 들고 있는 카드를 이름과 함께 출력한다.

        return:
            String, 예) {name}'s hand: {hand}

        """
        print("{}'s hand: {}".format(self.name, self.hand))

        return self

    def discard(self):
        """
        현재 들고 있는 카드중 마지막장을 반환하고, hand list에서 버린다.

        return:
            Card class

        """
        return self.hand.pop()

    def cal_hand_rank(self, manual_hands):
        """
        hand rank list에서 포커 랭크를 계산한다.
        기본 조건 3가지를 조합해서 포커의 모든 랭크를 계산할수 있다.
            . straight : 카드의 value가 순서대로 있는가?
            . flush : 카드의 suit가 같은 모양인가?
            . kinds : 같은 value의 카드가 몇장인가? 2장이면 one pair, 3장이면 three kinds 등

        return:
            Tuple, String : 예) (rank, kinds), rank name

                .Tuple의 첫번째 요소는 poker rank를 계산하여 표현한다(Royal straight flush는 10, One pair는 2)
                .Tuple의 두번째 요소는 kinds나 핸드를 표현한다. 3인 경우 3인 숫자가 2개 이상 있다는 뜻이다.
                    Straight, flush 등의 랭크는 같은 숫자의 중복으로 계산되지 않으므로, hand list를 저장한다.
                .Tuple의 세번째 요소는 두번째 요소에서 kinds를 저장했을경우, hand list를 저장한다.

                .String은 포커 랭크의 명칭을 출력한다.

        """

        # card의 value를 순서대로 정렬하기 위해 생성한다.
        # 1은, 14로 가장 큰수로 변환하고, 나머지는 그대로 둔다.
        ordered_values = ['.', '.', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '1']
        # ordered_values를 활용하여 dictionary를 만든다.문자 2는 숫자2, 문자 1은 숫자 14로 변환하기 위핸 dictionary이다.
        rank_values = dict((r, i) for i, r in enumerate(ordered_values))
        # hand list로 부터 value를 뽑아(String) 미리 만든 rank_dict에서 키와 매핑된 숫자 값을 찾고 내림차순으로 정렬한다.
        ranks = sorted([rank_values[str(item.value)] for item in manual_hands])
        ranks.reverse()

        # Straight를 판단하기 위한 최대 최소의 차와 모든 다른숫자인 경우로 맞으면 True를 반환한다
        # A,1,2,3,4,5는 최대 최소의 차로 판단할수 없는 경우로써 판단 기준을 추가함
        straight = ((max(ranks) - min(ranks)) == 4 and len(set(ranks)) == 5) or \
                   (len(set(ranks).intersection(set([14, 2, 3, 4, 5]))) == 5)

        # flush를 체크 하는 로직으로 모두 같은 문양이면 True를 반환한다.
        suits = [item.suit for item in manual_hands]
        flush = len(set(suits)) == 1

        # kind를 체크 하는 로직
        def kind(n, butnot=None):
            """
            rank 리스트에서 입력받은 n개 만큼 같은 숫자가 있으면 같은 숫자를 반환한다.

            Args:
                n : Integer, 같은 숫자가 몇개가 되는지 찾기위한 숫자
                butnot : 2 pair 인 경우, 찾은 숫자 말고, 다른 pair를 찾기 위해
                    지금 찾은 pair의 숫자를 제외하고 검색

            return:
                Integer, 예) n개의 같은 숫자의 수
                    .[10,5,5,3,2] kind(2)일 경우 5를 반환한다.
                    .[10,10,10,10,2] kind(4)일 경우 10를 반환한다.
            """
            for r in ranks:
                if ranks.count(r) == n and r != butnot:
                    return r
            return None

        # Straight 이고, flush이며, 카드의 숫자의 합이 60이면 Royal straight flush
        if straight and flush and sum(ranks) == 60:
            return (10, ranks), 'Royal straight flush'
        if straight and flush:
            return (9, ranks), 'Straight flush'
        # 4장의 같은 숫자가 있으면 Four of a kind
        if kind(4):
            return (8, kind(4), kind(1)), 'Four of a kind'
        if kind(3) and kind(2):
            return (7, kind(3), kind(2)), 'Full house'
        if flush:
            return (6, ranks), 'Flush'
        if straight:
            return (5, ranks), 'Straight'
        if kind(3):
            return (4, kind(3), ranks), 'Three of a kind'
        # 2개의 같은 숫자가 있고, 그 번호를 제외한 다른 2개의 같은 숫자가 있으면 Two paik
        if kind(2) and kind(2, kind(2)):
            return (3, kind(2), kind(2, kind(2)), ranks), 'Two pair'
        if kind(2):
            return (2, kind(2), ranks), 'One Pair'
        # 아무것도 없으면 High card
        return (1, ranks), 'High card'

    def rank(self, manual_hands=None):
        """
        hand list의 Card로 포커 랭킹을 계산한다.

        return:
            String,

        """
        if manual_hands is None:
            manual_hands = self.hand

        cal_rank = self.cal_hand_rank(manual_hands)
        print("{0}'s Hand Rank: <<{1}>>, {2}".format(self.name, cal_rank[1], cal_rank[0]))


class PokerRankTddTest(unittest.TestCase):
    """
    Unittest class
    Royal straight flush 부터 High card 까지 랭크를 테스트 한다

    """

    def test_royalsf(self):
        """
        Royal straight flush test

        """
        player1 = Player("Test")
        cards = list()
        cards.append(Card('Spades', 1))
        cards.append(Card('Spades', 13))
        cards.append(Card('Spades', 12))
        cards.append(Card('Spades', 11))
        cards.append(Card('Spades', 10))
        result = player1.cal_hand_rank(cards)
        self.assertEqual(result[0][0], 10, "test_royalsf got errors")

    def test_sf(self):
        """
        Straight flush test

        """
        player2 = Player("Test")
        cards = list()
        cards.append(Card('Hearts', 10))
        cards.append(Card('Hearts', 11))
        cards.append(Card('Hearts', 12))
        cards.append(Card('Hearts', 13))
        cards.append(Card('Hearts', 9))
        result = player2.cal_hand_rank(cards)
        self.assertEqual(result[0][0], 9, "test_sf got errors")

    def test_four_a_kind(self):
        """
        Four of a kind test

        """
        player3 = Player("Test")
        cards = list()
        cards.append(Card('Hearts', 10))
        cards.append(Card('Clubs', 10))
        cards.append(Card('Diamonds', 10))
        cards.append(Card('Spades', 10))
        cards.append(Card('Hearts', 5))
        result = player3.cal_hand_rank(cards)
        self.assertEqual(result[0][0], 8, "test_four_a_kind got errors")

    def test_full_house(self):
        """
        Full house test

        """
        player4 = Player("Test")
        cards = list()
        cards.append(Card('Hearts', 1))
        cards.append(Card('Clubs', 1))
        cards.append(Card('Diamonds', 9))
        cards.append(Card('Spades', 9))
        cards.append(Card('Hearts', 9))
        result = player4.cal_hand_rank(cards)
        self.assertEqual(result[0][0], 7, "test_full_house got errors")

    def test_flush(self):
        """
        Flush test

        """
        player4 = Player("Test")
        cards = list()
        cards.append(Card('Clubs', 10))
        cards.append(Card('Clubs', 9))
        cards.append(Card('Clubs', 8))
        cards.append(Card('Clubs', 7))
        cards.append(Card('Clubs', 4))
        result = player4.cal_hand_rank(cards)
        self.assertEqual(result[0][0], 6, "test_flush got errors")

    def test_straight(self):
        """
        Straight test

        """
        player4 = Player("Test")
        cards = list()
        cards.append(Card('Hearts', 1))
        cards.append(Card('Clubs', 2))
        cards.append(Card('Diamonds', 3))
        cards.append(Card('Spades', 4))
        cards.append(Card('Hearts', 5))
        result = player4.cal_hand_rank(cards)
        self.assertEqual(result[0][0], 5, "test_straight got errors")

    def test_three_of_a_kind(self):
        """
        Three of a kind test

        """
        player4 = Player("Test")
        cards = list()
        cards.append(Card('Hearts', 10))
        cards.append(Card('Clubs', 10))
        cards.append(Card('Diamonds', 10))
        cards.append(Card('Spades', 1))
        cards.append(Card('Hearts', 9))
        result = player4.cal_hand_rank(cards)
        self.assertEqual(result[0][0], 4, "test_three_of_a_kind got errors")

    def test_two_pair(self):
        """
        Two pair test

        """
        player4 = Player("Test")
        cards = list()
        cards.append(Card('Hearts', 10))
        cards.append(Card('Clubs', 10))
        cards.append(Card('Diamonds', 1))
        cards.append(Card('Spades', 1))
        cards.append(Card('Hearts', 3))
        result = player4.cal_hand_rank(cards)
        self.assertEqual(result[0][0], 3, "test_two_pair got errors")

    def test_one_pair(self):
        """
        One pair test

        """
        player4 = Player("Test")
        cards = list()
        cards.append(Card('Hearts', 2))
        cards.append(Card('Clubs', 2))
        cards.append(Card('Diamonds', 3))
        cards.append(Card('Spades', 10))
        cards.append(Card('Hearts', 8))
        result = player4.cal_hand_rank(cards)
        self.assertEqual(result[0][0], 2, "test_one_pair got errors")

    def test_high_card(self):
        """
        High test

        """
        player4 = Player("Test")
        cards = list()
        cards.append(Card('Hearts', 1))
        cards.append(Card('Clubs', 2))
        cards.append(Card('Diamonds', 5))
        cards.append(Card('Spades', 8))
        cards.append(Card('Hearts', 13))
        result = player4.cal_hand_rank(cards)
        self.assertEqual(result[0][0], 1, "test_high_card got errors")


if __name__ == "__main__":
    # Deck을 만들고, 섞는다
    myDeck = Deck()
    myDeck.shuffle()
    # 플레이어 객체의 이름을 JH, Paik로 만든고, 이름을 출력한다.
    player_paik = Player("JH, Paik")
    player_paik.sayHello()
    # Deck으로 부터 카드를 5장 받고, 받은카드 전부를 보여준다.
    for _ in range(10):
        num_of_card = 5
        player_paik.draw(myDeck, num_of_card)
        player_paik.showHand()
        # 포커 랭크를 계산하고 출력한다.
        player_paik.rank()
        print("----------------------------------------------------------------"
              "------------------------------------")
        for __ in range(num_of_card):
            player_paik.discard()

    # 테스트용으로 여러 포커의 랭크를 잘 계산하는지 확인한다.
    # unittest.main(verbosity=2)
