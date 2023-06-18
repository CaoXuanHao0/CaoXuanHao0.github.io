---
title:  "Generative Modeling for End-to-End Motion Prediction"
layout: post
mathjax: true
---

# Recap: What is End-to-End Motion Prediction
Traditionally the whole working pipeline of autonomous driving is modular pipeline:

![图片](/assets/blog2/modular_pipeline.jpg)

But it suffers from some drawbacks, like pice-wise training. So the recent trend is developing in an end-to-end fashion, which jointly develops perception and prediction modules:

![图片](/assets/blog2/e2e_motion_pred.jpg)

It has two key advantages: (1) shared computation and (2) shared information between modules.

The working pipeline of end-to-end motion prediction is as follows:

![图片](/assets/blog2/our_pipeline.jpg)

As you can see, there're three key components here, **perception model**, **conditional generative model**, and **decoder heads**. 

First, we have a perception model that encodes information into *past BEV feature maps* from past sensor input, and then we have a conditional generative model that generates *future BEV feature map* conditions on past BEV feature map (which we will elaborate on next), and then several heads are employed to decode them into future motion prediction.

The perception model is pretty much similar to those models for perception-only tasks like BEVFormer [], and actually, if you employ several decoder heads for the *past BEV feature maps*, you will get perception results, so I won't spend too much time talking about it. And the decoder heads are just usually detection heads, and segmentation heads, so we can directly use the existing model. The new thing here is the conditional generative model, which is borrowed from the prediction-only task but is modified a little. So in this blog I will focus on introducing the conditional generative model.

# Probabilistic prediction framework
We can model the prediction mechanism as a conditional probability distribution $$ P (Y |S) = p(y_1, · · ·, y_{T'} |s_1, · · ·, s_T ) $$ that generates T' future state feature $$ Y = \{y_1, y_2, · · ·, y_{T'} \} $$ given past state features for each frame $$ S = \{s_1, s_2, · · ·, s_T \} $$ or a single state feature $$ S=s $$ that represent all past frames, which are extracted from sensor data or/and HD maps.
Once we have an explicit form of this conditional probability distribution, we can sample future BEV features from it. 

![图片](/assets/blog2/gene.jpg)


But unfortunately, it's intractable, so we have to add some assumptions to factorize it.
Various factorizations of $$ P(Y | S) $$ utilize different levels of independence assumptions:

![图片](/assets/blog2/prob_pred.jpg)

**(a) Assume independent futures across time steps**, then we get:

 $$ P(Y |S) = \prod_{t}^{} P(y_t|S) $$
 
**(b) Autoregressive generation**

 $$ P(Y |S) = \prod_{t}^{} P(y_t|Y^{0:t-1},S) $$
 
The next trajectory depends on previous ones. We can model  $$ P(y_t \| Y^{0:t-1}, S) $$ as an RNN [] (then we're back to a deterministic model).

**(c) Assume there's a latent / hidden state that generates future trajectories [1]:**

$$ P (Y |S) = ∫ _Z P (Y |S, Z) P (Z|S) dZ $$

, where $$ Z $$ is a latent variable that captures all future (unobserved) scene dynamics (such as actor goals and style, multi-agent interactions, or future traffic light states).

But often modeling such probability is still intractable, so we still need to simplify it a little.
If we use a deterministic mapping $$ Y = f (S, Z) $$ to implicitly characterize $$ P (Y |S, Z) $$ instead of explicitly representing it in a parametric form, we can avoid factorizing $$ P (Y |S, Z) $$. In this framework, generating scene-consistent future state $$ Y $$ (by sample $$Y$$ from $$ P(Y|S) $$) is simple and highly efficient since it only requires one stage of parallel sampling:

  Step 1. Draw latent scene samples from prior $$ Z ∼ P (Z \|S) $$ (equivalent to the sample from a "future distribution [2]" that we'll discuss later)
  
  Step 2. Decode with the deterministic decoder $$ Y = f_{predict} (S, Z) \approx P (Y \|S, Z) $$.
  
This is exactly the same as any other deterministic end-to-end motion prediction mechanism $$ Y=f_{predict}(X) $$, with an additional probabilistic element $$ Z $$ added, which captures all stochasticity in the generative process.

# Future Distributions (w and w / o GT)
To learn a distribution about future latent state $$Z$$, we formulate two future distributions with and without GT future state $$ Y_{GT}=(y_{t+1}^{GT}, ..., y_{t+T'}^{GT} ) $$: $$ P(Z|S) $$ and $$ P(Z|S, Y_{GT}) $$, where $$ P(Z|S) $$ is what we actually use for inference (cover all the possible modes contained in the future) and need to be learned, while $$ P(Z|S, Y_{GT}) $$ additionally take ground truth future state as input and thus is used as supervision for learning $$ P(Z|S) $$.

We parametrize both distributions as diagonal Gaussians with mean $$ μ \in R^L $$ and variance $$ σ^2 \in R^L $$ which are learnable parameters, $$L$$ being the latent dimension: $$ P(Z \| S) = N (μ_{t}, σ^2_{ t}) $$, $$ P(Z | S,Y_{GT}) = N (μ_{t,gt}, σ^2_{ t,gt}) $$.

**(1) Future distribution w/o GT:$$ P(Z|S) $$**: represents what could happen given the past context 
* Input past state feature $$ S $$ that represents the past T frames (as the condition of distribution).
* Transformed $$S$$ by a learnable NN (map dim of $$S$$ to the desired latent dimension $$L$$).
* Downsampling convolutional layers + average pooling layer + FC
* Two heads for outputing parametrization of the present distribution: $$ (μ_{t}, σ_{t}) \in R^L × R^L $$.

**(2) Future distribution w GT: $$ P(Z|S,Y_{GT}) $$**: represents what actually happened in that particular observation.
* Input $$ S $$, and information about future frames (Gt label/state/latent of future prediction task $$ Y_{GT} $$, "observed future")
* Transformed $$S$$ and $$Y_{GT}$$ by a learnable NN.
* For $$S$$: Downsampling convolutional layers + average pooling + FC as before.
* Concatenate output of all time steps, and feed them to another FC.
* Output parametrisation of the present distribution: $$ (μ_{t,gt}, σ_{t,gt}) \in R^L × R^L $$.

And then we learn by encouraging $ P(Z|S) $to mach $ P(Z|S, Y_{GT}) $by optimizing the following loss (min the KL divergence between two distributions) (or equivalently, optimize ELBO):
$$\text{min} \ D_{KL}(P(S|Z) ||P(S|Z,Y_{GT}) )$$


**Probabilistic Future Prediction**

During training: sample latent future from $$ P(Z|S,Y_{GT}) $$: $$ Z ∼ N (μ_{t,gt}, σ^2_{ t,gt}) $$, for all future frames prediction.

During inference: sample latent future from $$ P(Z|S) $$:  $$ Z ∼ N (μ_{t}, σ^2_{ t}) $$ (each sample corresponds to a different possible future).

**Making Future Prediction**
So the whole process can be simplified as follows:
(1) Sample latent code $$ Z $$  from $$ P(Z|S) $$, and then (2) feed $$ Z $$ (encode probabilistic future information) and $$ S $$ (encode information of current and past frames) into decoder $$ f_{predict} $$ (usually an RNN), to generate future features $$ \{ y_{t+j} \} _ {j \in J} = f_{predict}(S, Z) $$ that (3) become the inputs of several individual heads to decode these features to downstream prediction task:

![图片](/assets/blog2/prob_simple.jpg)


# Reference

[1] Implicit Latent Variable Model for Scene-Consistent Motion Forecasting.

[2] Probabilistic Future Prediction for Video Scene Understanding.

[3] 

[4] 

