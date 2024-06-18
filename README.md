# Baseline model for VLSP 2023 ComOM Shared Task

## Task
This task aims to create models that can find opinions from product reviews. Each review has sentences that compare different parts of products.
<p align="center">
<img src="image/table1.png" width="50%" />
</p>

## Dataset
The dataset is released by VLSP 2023 challenge on
Comparative Opinion Mining from Vietnamese Product
Reviews. Each review contains comparative sentences,
and the corresponding quintuples are annotated.
The following table shows the statistics of the comparative quintuple corpora.
<p align="center">
<img src="image/table2.png" width="50%" />
</p>

## Approach
### Overall Architecture
<p align="center">
<img src="image/baseline_model.png" width="50%" />
</p>


### Stage 1: Comparative Element Extraction + Comparative Sentence Identification
<p align="center">
<img src="image/stage1.png" width="50%" />
</p>

### Stage 2, 3: Combination, Filtering + Comparision Label Classification
**Combination**
<p align="center">
<img src="image/stage2_combi.png" width="50%" />
</p>

**Filtering**
<p align="center">
<img src="image/stage2_filter.png" width="50%" />
</p>

**Comparision Label Classification**
<p align="center">
<img src="image/stage2_clc.png" width="50%" />
</p>

**Output**
<p align="center">
<img src="image/stage2_output.png" width="50%" />
</p>


