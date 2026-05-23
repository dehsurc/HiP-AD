
## 전체 연구 방향성
- 단순히 module별로 teacher를 두고 Multi-module distillation하기
    - Contribution이 부족할 수 있음
    - “KD를 했더니 성능이 올랐다” 정도의 메세지가 전부
    - 이 경우 distillation experiment를 정말 흠 없이 튼튼하게 준비하는게 필요할수도.
        - ~~(물론 다른 실험에서도 튼튼하게 하는 건 필요)~~
- 그럼 distillation 왜 해야하는데?
    - 기존 distillation method에서 주장하는 장점정도는 언급 가능(e.g. 효율성, 자원 절약? 등.. 뭐있냐)
        - 그러나 이정도 수준의 주장은 어림도 없다
- End-to-end Driving model 고찰
    - modular E2E 모델이 필요한 이유는?
        - Planning이 실패하는 사례에 대해 module 별 성능 분석이 가능하기 때문
    - module 별 향상이 필요한 이유는?
        - 기본적으로 ‘**모듈 성능이 오르면 planning 성능도 오른다**’라는 명제가 성립한다는 가정이 들어감
        - 그러나 이는 명시적인 증명이 없는 이상 무의미한 작업일지도 모른다.
            - 어쩌면 planning 성능이 오르더라도..
        - 명시적 증명이란?
            - **planning을 잘할 수 있도록 하는 모듈 성능**의 향상
    - 확실한 메세지가 필요한 시점
        - scene마다 object detection이 중요할수도, map detection이 중요할수도. 혹 둘 다 필요하거나 반대인 경우도 있을 것이다
        - 단순히 distillation을 다 걸어주는 구조에서 마무리짓는게 아니라, 장면마다 중요한 task의 중요도를 올려주는 방식을 채택하자
    - 다시 말해, scene마다 task간 중요도가 다르기 때문에 distillation도 scene에 최적화 시켜주는 과정이 필요하다
        - det, map, motion 등 현재 scene에서 어떤 task가 중요한지 어케 아냐
            - 명시적으로 annotate는 어렵다(=직접적 causality 부여는 불가능)
            - 결국, 궁극적으로 E2E model은 **planning을 잘하는 것이 목적**이다.
            - **gradient alignment** 활용을 통해 간접적으로 추정이 가능하다고 주장(=planning loss와 같은 방향으로 task loss가 간다면 이는 중요한 task로 분류)
    - 근데, 그래서 왜 distillation인데?
        - module-wise teacher는 object, map, agent motion 등 driving scene factor에 대해서 modular e2e model의 module 별 성능을 상회한다
        - 이 factor들을 더 잘 이해한다는 것은 scene representation을 더 잘한다고 볼 수 있다.
            - 물론, 이것이 planning을 더 잘하기 위해 무조건 필요한 능력은 아니다.
        - knowledge distillation을 통해 module-wise teacher의 scene representation 능력을 쉽게 가져오자
            - 다만 모든 상황에 대해 배우자는게 아니다. planning을 잘 하기 위해 도움이 되는 representation만 잘 배우자는 것(Planning-Aligned Modular Distillation)
            - 단순히 output loss뿐만 아니라 latent space에서도 배울 수 있을 것(feature-distillation)
        - 이 때도 Real GT로 배우는 module GT loss도 gradient alignment 해야 하는것 아니냐,  또는 이걸 맞춰주면 되지 왜 distill이냐. 할 수 있음
            - 이거도 해보고 비교해봐야지 뭐..
            - 혹은 logit-based가 아닌 feature distill 등을 통해 표현력을 배운다고 주장
                - output loss보다는 feature 에서는 더 많은 정보가 있을 것
                - 명분을 살리려면 feature kd를 많이 쓰던가 해야 함 

## 1 step probe 분석을 통해 알고싶은것
1. 기존 v3에서 하던 것처럼 task 간의 loss 변화율 분석
2. planning sensitivity 분석 보완
3. planning에게 도움을 주는 task가 무엇인지 어떻게 판단할 수 있을지 고민해줘.