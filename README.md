# transformer-based-meta-matching-stacking


## Introduction

- 유동성 지능(Fluid intelligence)은 새로운 상황에서 논리적인 추론을 통해 문제를 해결하는 능력으로써 인간의 보편적 지성의 중요한 요소입니다.
- 뇌 영상 데이터 및 비영상 표현형(non-imaging phenotype)이 통합된 데이터 셋의 증가는 신경과학의 예측 모렐링의 발전에 기여하였습니다.
- Meta-matching : 대규모 데이터 셋으로 학습된 모델을 소규모 데이터 셋의 비영상 형질의 예측에 사용하는 프레임워크이지만, 다양한 모달리티의 정보를 통합하고 활용하지 못합니다.
- 본 연구에서는 self-attention을 통해 모달리티 간 비영상 표현형의 유의미한 관게성을 학습하고 유동성 지능 예측에 사용하는 multimodal transformer stacking 기법을 제안합니다.

## Methods

### Dataset

- Human Connectome Project (HCP)에서 제공하는 59개의 비영상 표현형과 750명의 뇌 영상 데이터를 사용하였습ㄴ다.
- 학습에 사용하는 모달리티 별 이미지 특징 데이터입니다.
    - **sMRI** : morphometric inverse divergence networks (MIND)
    - **fMRI** : 휴지기(resting state)와 일시적인 기억(working memory)과 관련된 작업에 대한 기능적 연결성(functional connectivity)
    - **dMRI** : Tract-Based Spatial Statics (TBSS)와 tractography에서의 주축 방향 확산도(axial diffusivity, AD)
- Train meta set과 test meta set은 8:2의 비율로 분할되며, test meta set은 다시 100명의 k-shot sample set과 50명의 test set으로 분할하였습니다.
- Train meta set은 유동성 지능을 제외한 58개의 비영상 표현형을, test meta set은 유동성 지능을 포함합니다.

### Multimodal Transformer Stacking

- Train meta set을 통해 Basic DNN을 학습합니다.
- K-shot sample set의 5종류의 이미지 특징 데이터에서 학습된 Basic DNN을 통해 58개의 비영상 표현형을 예측합니다.
- 각 이미지 특징 데이터로부터 나온 58개의 예측 값을 병합하여 multimodal feature matrix를 생성합니다.
- Linear Layer를 거쳐 multimodal feature matrix를 transformer encoder에 적합한 차원으로 변환합니다.
- Matrix를 9개의 layer를 가진 transformer encoder에 입력으로 넣어 self-attentinon을 통해서 비영상 표현형 간 관계를 학습하도록 합니다.
- Transformer encoder로부터 나온 matrix를 1차원 벡터로 변환한 이후 dense layer를 거쳐 유동성 지능을 예측합니다.

### Experiments

- 평가 지표 : Pearson’s Correlation (r), Coefficient of Determination (COD)
- Train meta set과 test meta set의 subject 구성을 다르게 하여 50번의 반복 실험을 통해 성능 분포를 형성합니다.
- Multimodal transformer stacking과 실험에서 사용된 각 이미지 특징 데이터의 advanced stacking 성능을 비교하였습니다.
- Multimodal transformer stacking과 5가지 이미지 특징 데이터에 대해 advanced stacking의 예측을 평균하여 최종 예측을 수행하는 stacking average 성능을 비교합니다.

## Results

![Figure 2](/figure/Final_figure_1.png)

- 단일 이미지 특징 데이터를 활용한 advanced stacking 결과 중에서 가장 성능이 우수한 working memory advanced stacking에 비해 multimodal transformer stacking에서 평균 correlation 0.07, 평균 COD 0.05가 향상된 결과를 보였습니다.
- Average Stacking 방법에 대비 multimodal transformer stacking의 평균 correlation은 0.12가 향상되었고, 평균 COD는 0.08 향상되었습니다.
- Multimodal transformer stacking과 기존 방법에 대한 성능 비교를 통해, 멀티모달리티로부터 추출된 정보를 self-attnetion을 활용하여 효과적으로 융합할 수 있음을 보였습니다.

## Conclusion

- 본 연구에서 유동성 지능의 정확한 예측을 위해 다양한 모달리티의 정보를 학습할 수 있는 새로운 stacking 기법인 multmodal transformer stacking 기법을 제안하였습니다.
- Multimodal transformer stakcing이 유동성 지능을 예측하는 실험에서 가장 우수한 예측 성능을 보였고, 다른 실험들과 통계적으로 유의미한 차이가 있음을 보였습니다.
- Multimodal transformer stacking에서 서로 다른 모달리티의 비영상 표현형 간 관계를 파악하는 transformer encoder의 능력이 유동성 지능 예측 성능 향상에 기여하였습니다.

## Reference

[1] He, T., An, L., Chen, P., Chen, J., Feng, J., Bzdok, D., Holmes, A. J., Eickhoff, S. B., & Yeo, B. T.
(2022). 'Meta-matching as a simple framework to translate phenotypic predictive models from big
to small data', Nature neuroscience, vol. 25, no.6, pp. 795-804.
[2] Bahdanau, D., Cho, K., & Bengio, Y. (2014). 'Neural machine translation by jointly learning to align
and translate', arXiv preprint arXiv:1409.0473, vol., pp.
[3] Ooi, L. Q. R., Chen, J., Zhang, S., Kong, R., Tam, A., Li, J., Dhamala, E., Zhou, J. H., Holmes, A. J.,
& Yeo, B. T. (2022). 'Comparison of individualized behavioral predictions across anatomical, diffusion
and functional connectivity MRI', Neuroimage, vol. 263, pp. 119636.
[4] Sebenius, I., Seidlitz, J., Warrier, V., Bethlehem, R. A., Alexander-Bloch, A., Mallard, T. T., Garcia,
R. R., Bullmore, E. T., & Morgan, S. E. (2023). 'Robust estimation of cortical similarity networks from
brain MRI', Nature neuroscience, vol. 26, no.8, pp. 1461-1471.
[5] Seidlitz, J., Váša, F., Shinn, M., Romero-Garcia, R., Whitaker, K. J., Vértes, P. E., Wagstyl, K.,
Reardon, P. K., Clasen, L., & Liu, S. (2018). 'Morphometric similarity networks detect microscale
cortical organization and predict inter-individual cognitive variation', Neuron, vol. 97, no.1, pp.
231-247. e237.
[6] Schaefer, A., Kong, R., Gordon, E. M., Laumann, T. O., Zuo, X.-N., Holmes, A. J., Eickhoff, S. B., &
Yeo, B. T. (2018). 'Local-global parcellation of the human cerebral cortex from intrinsic functional
connectivity MRI', Cerebral cortex, vol. 28, no.9, pp. 3095-3114.V
[7] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin,
I. (2017). ‘Attention is all you need’, Advances in neural information processing systems, vol. 30,
pp.
