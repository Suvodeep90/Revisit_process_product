# Revisiting Process versus Product Metrics: a LargeScale Analysis

Numerous  methods can  build predictive models from software  data. But what methods and conclusions should we endorse as we move from analytics in-the-small (dealing with a handful of projects) to analytics in-the-large (dealing with hundreds of projects)? 

To answer this question, we recheck prior small-scale results (about process versus product metrics for defect prediction and the granularity of metrics) using 722,471 commits from 700 Github projects. We find that some analytics in-the-small conclusions still hold  when scaling up to analytics in-the-large.  For example, like prior work,  we see that  process metrics are better predictors for defects than product metrics (best process/product-based learners respectively achieve   recalls of 98%/44% and AUCs of 95%/54%, median values). 

That said,  we warn that it is unwise to trust metric importance results from analytics in-the-small studies since those change, dramatically when moving  to analytics in-the-large. Also, when reasoning in-the-large about hundreds of projects, it is better to use predictions from multiple models (since single model predictions can become confused and  exhibit a high variance).



