# CheckHateKorean
한국어 혐오 표현 판별 인공지능

혐오 발언 감지 인공지능은 인공지능이 빠르게 혐오 발언을 필터링하고 이를 사람이 한 번 더 확인하게 함으로써 사람이 실제로 하는 일을 대폭 줄여주기 위해 만들어졌다.

이 인공지능의 아이디어는 넥슨 코리아 인텔리전스랩스의 어뷰징 탐지팀의 발표에서 CNN을 통해 욕설을 탐지한다는 점이 흥미로워 이를 참고해 개발하였다. 
문장의 모든 단어를 초성, 중성, 종성으로 나눈 뒤 이를 인코딩 하여 CNN과 LSTM을 이용해 각 글자의 초성 중성 종성의 조합을 살펴보면서 글자 사이의 조합 또한 고려할 수 있도록 하였다.

초반에는 CNN만을 이용해 구성했으나 기대한 만큼의 정확도가 나오지 않았고(대략 0.3에서 0.4 정도) output 수치가 고정되는 일도 많아서 LSTM을 첨가해 보았다

kocohub의 korean-hate-speech 데이터셋을 사용하였으며 학습 후 테스트 해본 결과 
24살인데 술을먹어 발랑까졋네 고딩때도 먹엇을듯 hate
안녕, 반가워 none
으로 첫 번째 문장은 혐오 발언으로 두 번째 문장은 일반적인 발언으로 판단했다.

또한 욕설의 경우에서도 
씨발 hate
씨12발 hate
처럼 어느정도 변형된 경우도 판단할 수 있었다.