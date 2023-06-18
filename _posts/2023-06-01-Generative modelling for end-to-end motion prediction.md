---
title:  "Generative modeling for end-to-end motion prediction"
layout: post
mathjax: true
---

Probabilistic prediction framework
We can model the prediction mechanism as a conditional probability distribution $$ P (Y |S) = p(y_1, · · · , y_{T'} |s_1, · · · , s_T ) $$ over T' future state feature $$ Y = \{y_1, y_2, · · · , y_{T'} \} $$ given past state features for each frame $$ S = \{s_1, s_2, · · · , s_T \} $$ or a single state feature $$ S=s $$ that represent all past frames, which are extracted from sensor data or/and HD maps.


Various factorizations of $ P(Y | S) $ utilize different levels of independence assumptions:
(dash line means it does not explicit appear in the formulation)

**(a) Assume independent futures across time steps**, then we get:

 $$ p(Y |S) = \prod_{t}^{} p(y_t|S) $$
 
**(b) Autoregressive generation**

 $$ p(Y |S) = \prod_{t}^{} p(y_t|Y^{0:t-1},S) $$
 
The next trajectory depends on previous ones. We can model  $$ p(y_t|Y^{0:t-1},S) $$ as an RNN [] (then we're back to a deterministic model).

**(c) Assume there's a latent / hidden state that generates future trajectories [1]:**

$ P (Y |S) = ∫ _Z P (Y |S, Z) P (Z|S)dZ $

, where $$ Z $$ is a latent variable that captures all future (unobserved) scene dynamics (such as actor goals and style, multi-agent interactions, or future traffic light states).

But often modeling such probability is still intractable, so we still need to simplify it a little.
If we use a deterministic mapping $$ Y = f (S, Z) $$ to implicitly characterize $$ P (Y |S, Z) $$ instead of explicitly representing it in a parametric form, we can avoid factorizing $$ P (Y |S, Z) $$. In this framework, generating scene-consistent future state $$ Y $$ (by sample $$Y$$ from $$ P(Y|S) $$) is simple and highly efficient since it only requires one stage of parallel sampling:
  Step 1. Draw latent scene samples from prior $$ Z ∼ P (Z|S) $$ (equivalent to the sample from a future distribution that we'll discuss later)
  Step 2. Decode with the deterministic decoder $$ Y = f_{predict} (S, Z)\approx P (Y |S, Z) $$.
This is exactly the same as any other end-to-end motion prediction mechanism $$ Y=f_{predict}(X) $$, with an additional probabilistic element $$ Z $$ added, which captures all stochasticity in the generative process.

Future Distributions (w and w / o GT)
To learn a distribution about future latent state $$Z$$, we formulate two future distributions with and without GT future state $$ Y_{GT}=(y_{t+1}^{GT}, ..., y_{t+T'}^{GT} ) $$: $$ P(Z|S) $$ and $$ P(Z|S,Y_{GT}) $$, where $$ P(Z|S) $$ is what we actually use for inference (cover all the possible modes contained in the future) and need to be learned, while $$ P(Z|S,Y_{GT}) $$ additionally take ground truth future state as input and thus is used as supervision for learning $$ P(Z|S) $$.

We parametrize both distributions as diagonal Gaussians with mean $$ μ \in R^L $and variance $$ σ^2 ∈ R^L $$ which are learnable, $$L$$ being the latent dimension: $$ P(Z|S) = N (μ_{t}, σ^2_{ t}) $$, $$ P(Z|S,Y_{GT}) = N (μ_{t,gt}, σ^2_{ t,gt}) $$

(1) Future distribution w/o GT:$$ P(Z|S) $$: represents what could happen given the past context 
Input past state feature数学公式: $ S $that represents the past T frames (as the condition of distribution).
Transformed S by a learnable NN (map dim of S to the desired latent dimension L).
Downsampling convolutional layers + average pooling layer + FC
Two heads for outputing parametrization of the present distribution: $$ (μ_{t, present}, σ_{t, present}) \in R^L × R^L $$.
(2) Future distribution w GT: $$ P(Z|S,Y_{GT}) $$: represents what actually happened in that particular observation.
Input $$ S $$, and information about future frames (Gt label/state/latent of future prediction task $$ Y_{GT} $$, "observed future")
Transformed $$S$$ and $$Y_GT$$ by a learnable NN
For $$S$$: Downsampling conv + average pooling + FC as before
Concatenate output of all t, feed them to another FC
Output parametrisation of the present distribution: $$ (μ_{t,future}, σ_{t,future}) \in R^L × R^L $$.

And then we learn by encouraging $ P(Z|S) $to mach $ P(Z|S,Y_{GT}) $by optimizing the following loss (min the KL divergence between two distributions):
(or equivalently, optimize ELBO)

* **Probabilistic Future Prediction**
During training: sample latent future from $$ P(Z|S,Y_{GT}) $$: $$ Z ∼ N (μ_{t,gt}, σ^2_{ t,gt}) $$, for all future frames prediction.

During inference: sample latent future from $$ P(Z|S) $$:  $$ Z ∼ N (μ_{t}, σ^2_{ t}) $$ (each sample corresponds to a different possible future).

* **Making Future Prediction**
Feed $$ Z $$ (encode probabilistic future information) and $$ S $$ (encode information of current and past frames) into decoder $ f_{predict} $(usually an RNN), to sequentially generate future features $$ \{ y_{t+j} \}_{j \in J} = f_{predict}(S,Z) $$ that become the inputs of several individual heads D to decode these features to downstream prediction task.

Reference
[1] Implicit Latent Variable Model for Scene-Consistent Motion Forecasting.
[2] Probabilistic Future Prediction for Video Scene Understanding.

