# Medical Vision and NLP at millenium iHEALTH

Denis Parra, CS Professor, PUC Chile

Cecilia Besa, Med Professor, PUC Chile

Jocelyn Dunstan, CS Professor, UChile

Pablo Messina, PhD(c)

Ivania Donoso, PhD

Tamara Quiroga, PhD

Gregory Schuit, MSc Student, PUC Chile

Jorge P

Ricardo Schilling, MSc Student, PUC Chile

**Note**: __Keep it sorted by year__

## Papers

### Report Generation
* (method) Liu, C. F., Zhao, Y., Miller, M. I., Hillis, A. E., & Faria, A. (2022). Automatic comprehensive radiological reports for clinical acute stroke MRIs. Available at SSRN 4123512. [[pdf]](https://assets.researchsquare.com/files/rs-1705683/v1_covered.pdf?c=1654013465) [[exe]](https://www.nitrc.org/doi/landing_page.php?table=frs_file&id=12723)

* (medical paper) Brady, A. (2022). Language and Radiological Reporting. In Structured Reporting in Radiology (pp. 1-19). Springer, Cham.

* (method RATCHET) Hou, B., Kaissis, G., Summers, R. M., & Kainz, B. (2021, September). RATCHET: Medical Transformer for Chest X-ray Diagnosis and Reporting. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 293-303). Springer, Cham. [[pdf]]
(https://arxiv.org/abs/2107.02104) [[code]](https://github.com/farrell236/RATCHET)

* (method MedViLL) Moon, J. H., Lee, H., Shin, W., & Choi, E. (2021). Multi-modal Understanding and Generation for Medical Images and Text via Vision-Language Pre-Training. arXiv preprint arXiv:2105.11333. [[pdf]](https://arxiv.org/pdf/2105.11333.pdf) [[code & data]](https://github.com/SuperSupermoon/MedViLL)

* (survey)  Messina, P., Pino, P., Parra, D., Soto, A., Besa, C., Uribe, S., ... & Capurro, D. (2022). A survey on deep learning and explainability for automatic report generation from medical images. ACM Computing Surveys (CSUR).

* (method) Yuan, J., Liao, H., Luo, R., & Luo, J. (2019, October). Automatic radiology report generation based on multi-view image fusion and medical concept enrichment. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 721-729). Springer, Cham. [[pdf]](https://arxiv.org/pdf/1907.09085.pdf) 

* (method) Xue, Y., Xu, T., Rodney Long, L., Xue, Z., Antani, S., Thoma, G. R., & Huang, X. (2018, September). Multimodal recurrent model with attention for automated radiology report generation. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 457-466). Springer, Cham. [[pdf]](https://faculty.ist.psu.edu/suh972/Xue-MICCAI2018.pdf)

* (method) Miura, Yasuhide and Zhang, Yuhao and Tsai, Emily Bao and Langlotz, Curtis P. and Jurafsky, Dan. (2020). Improving Factual Completeness and Consistency of Image-to-Text Radiology Report Generation. arXiv preprint arXiv:2010.10042. [[pdf]](https://arxiv.org/pdf/2010.10042.pdf) [[code]](https://github.com/ysmiura/ifcc)

#### Medical VQA

* (method MedViLL) Moon, J. H., Lee, H., Shin, W., & Choi, E. (2021). Multi-modal Understanding and Generation for Medical Images and Text via Vision-Language Pre-Training. arXiv preprint arXiv:2105.11333. [[pdf]](https://arxiv.org/pdf/2105.11333.pdf) [[code & data]](https://github.com/SuperSupermoon/MedViLL)

* (method CPRD) Liu, B., Zhan, L. M., & Wu, X. M. (2021, September). Contrastive Pre-training and Representation Distillation for Medical Visual Question Answering Based on Radiology Images. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 210-220). Springer, Cham. [[pdf]](http://www4.comp.polyu.edu.hk/~csxmwu/papers/MICCAI-2021-Med_VQA.pdf)

* (method MMQ-VQA) Do, T., Nguyen, B. X., Tjiputra, E., Tran, M., Tran, Q. D., & Nguyen, A. (2021, September). Multiple meta-model quantifying for medical visual question answering. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 64-74). Springer, Cham. [[code]](https://github.com/aioz-ai/MICCAI21_MMQ)

* Eslami, S., de Melo, G., & Meinel, C. (2021). Does CLIP Benefit Visual Question Answering in the Medical Domain as Much as it Does in the General Domain?. arXiv preprint arXiv:2112.13906. [[pdf]](https://paperswithcode.com/paper/2112-13906)

* Vu, M. H., Löfstedt, T., Nyholm, T., & Sznitman, R. (2020). A question-centric model for visual question answering in medical imaging. IEEE transactions on medical imaging, 39(9), 2856-2868. [[code]](https://github.com/vuhoangminh/vqa_medical) -> relies heavely on MUTAN for visual-text fusion scheme [[code]](https://github.com/Cadene/vqa.pytorch)

* (method QCR) Zhan, L. M., Liu, B., Fan, L., Chen, J., & Wu, X. M. (2020, October). Medical visual question answering via conditional reasoning. In Proceedings of the 28th ACM International Conference on Multimedia (pp. 2345-2354). [[pdf]](http://www4.comp.polyu.edu.hk/~csxmwu/papers/MM-2020-Med-VQA.pdf) [[code]](https://github.com/awenbocc/med-vqa)

* (method CGMVQA) Ren, F., & Zhou, Y. (2020). Cgmvqa: A new classification and generative model for medical visual question answering. IEEE Access, 8, 50626-50636. [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9032109) [[code]](https://github.com/youngzhou97qz/CGMVQA)

* (method MEVF) Nguyen, B. D., Do, T. T., Nguyen, B. X., Do, T., Tjiputra, E., & Tran, Q. D. (2019, October). Overcoming data limitation in medical visual question answering. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 522-530). Springer, Cham.
[[pdf]](https://arxiv.org/abs/1909.11867) [[code]](https://github.com/aioz-ai/MICCAI19-MedVQA) [[data]](https://www.nature.com/articles/sdata2018251#data-citations)

* Many medical VQA works are based on [[Bilinear Attention Networks]](https://arxiv.org/abs/1805.07932) and its implemented pytorch [[code]](https://github.com/jnhwkim/ban-vqa)

#### XAI

* Weber, L., Lapuschkin, S., Binder, A., & Samek, W. (2022). Beyond Explaining: Opportunities and Challenges of XAI-Based Model Improvement. arXiv preprint arXiv:2203.08008. [[pdf]](https://arxiv.org/pdf/2203.08008.pdf)

* Lee, K. H., Park, C., Oh, J., & Kwak, N. (2021). LFI-CAM: Learning Feature Importance for Better Visual Explanation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 1355-1363). [[code]](https://github.com/TrustworthyAI-kr/LFI-CAM)

* (NLP) Jain, S., & Wallace, B. C. (2019, June). Attention is not Explanation. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) (pp. 3543-3556).

* (counterfactual) Graham Spinks, Marie-Francine Moens. Justifying diagnosis decisions by deep neural networks. Journal of Biomedical Informatics
Volume 96, August 2019, 103248: https://www.sciencedirect.com/science/article/pii/S1532046419301674

#### Text Groundig for Visual Tasks
* Petryk, S., Dunlap, L., Nasseri, K., Gonzalez, J., Darrell, T., & Rohrbach, A. (2022). On Guiding Visual Attention with Language Specification. arXiv preprint arXiv:2202.08926.

* Ross, A. S., Hughes, M. C., & Doshi-Velez, F. (2017, January). Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations. In IJCAI.

#### Multimodal & Contrastive Learning

* Taleb, A., Kirchler, M., Monti, R., & Lippert, C. (2021). ContIG: Self-supervised Multimodal Contrastive Learning for Medical Imaging with Genetics. arXiv preprint arXiv:2111.13424.

* (method) Eslami, S., de Melo, G., & Meinel, C. (2021). Does CLIP Benefit Visual Question Answering in the Medical Domain as Much as it Does in the General Domain?. arXiv preprint arXiv:2112.13906. [[pdf]](https://paperswithcode.com/paper/2112-13906) [[code]](https://github.com/sarahESL/PubMedCLIP/tree/main/PubMedCLIP)

* (method CPRD) Liu, B., Zhan, L. M., & Wu, X. M. (2021, September). Contrastive Pre-training and Representation Distillation for Medical Visual Question Answering Based on Radiology Images. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 210-220). Springer, Cham. [[pdf]](http://www4.comp.polyu.edu.hk/~csxmwu/papers/MICCAI-2021-Med_VQA.pdf)

* (method CMSA) Gong, H., Chen, G., Liu, S., Yu, Y., & Li, G. (2021, August). Cross-Modal Self-Attention with Multi-Task Pre-Training for Medical Visual Question Answering. In Proceedings of the 2021 International Conference on Multimedia Retrieval (pp. 456-460). [[pdf]](https://dl.acm.org/doi/abs/10.1145/3460426.3463584) [[code]](https://github.com/haifangong/CMSA-MTPT-4-MedicalVQA)

* (method PCRL) Zhou, H. Y., Lu, C., Yang, S., Han, X., & Yu, Y. (2021). Preservational Learning Improves Self-supervised Medical Image Models by Reconstructing Diverse Contexts. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 3499-3509). [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_Preservational_Learning_Improves_Self-Supervised_Medical_Image_Models_by_Reconstructing_Diverse_ICCV_2021_paper.pdf) [[code]](https://github.com/Luchixiang/PCRL)

#### Graph Neural Networks & Knowledge-Aware Methods

* Agu, N. N., Wu, J. T., Chao, H., Lourentzou, I., Sharma, A., Moradi, M., ... & Hendler, J. (2021, September). Anaxnet: Anatomy aware multi-label finding classification in chest x-ray. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 804-813). Springer, Cham.

#### NLP Español

* Cotik, V., Stricker, V., Vivaldi, J., & Rodríguez Hontoria, H. (2016). Syntactic methods for negation detection in radiology reports in Spanish. In Proceedings of the 15th Workshop on Biomedical Natural Language Processing, BioNLP 2016: Berlin, Germany, August 12, 2016 (pp. 156-165). Association for Computational Linguistics.

* Stricker, V., Iacobacci, I., & Cotik, V. (2015). Negated findings detection in radiology reports in spanish: an adaptation of negex to spanish. In Ijcai-workshop on replicability and reproducibility in natural language processing: adaptative methods, resources and software, buenos aires, argentina.

## Datasets
* [VQA-RAD dataset](https://www.nature.com/articles/sdata2018251): A manually constructed VQA dataset in radiology. 315 images and 3515 visual questions. 104 head axial single-slice CTs or MRIs, 107 chest x-rays, and 104 abdominal axial CTs. The VQA-RAD test set contains 151 matched pairs of free-form and paraphrased questions. [Download link](https://osf.io/89kps/)

* [SLAKE](https://www.med-vqa.com/slake/): A Semantically-Labeled Knowledge-Enhanced Dataset for Medical Visual Question Answering

* MIMIC-CXR

* Open IU

* Chest ImaGenome: A joint rule-based natural language processing (NLP) and CXR atlas-based bounding box detection pipeline are used to automatically label 242072 frontal MIMIC CXRs locally. Contributions:  1) 1,256 combinations of relation annotations between 29 CXR anatomical locations (objects with bounding box coordinates) and their attributes, structured as a scene graph per image, 2) over 670,000 localized comparison relations (for improved, worsened, or no change) between the anatomical locations across sequential exams, as well as 3) a manually annotated gold standard scene graph dataset from 500 unique patients. [[data]](https://www.physionet.org/content/chest-imagenome/1.0.0/) [[pdf]](https://arxiv.org/pdf/2105.09937.pdf)

* CXR Eye Gaze: Creation and Validation of a Chest X-Ray Dataset with Eye-tracking and Report Dictation for AI Tool Development [code] https://github.com/cxr-eye-gaze/eye-gaze-dataset

* RadReport Template Library https://radreport.org/home/232/2012-07-12%2000:00:00


## Code
**MEVF** (2019)  [code](https://github.com/aioz-ai/MICCAI19-MedVQA)
