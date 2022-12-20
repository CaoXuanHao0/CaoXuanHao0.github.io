Adversarial generalisation

It's a ML technique for "preparing and generalizing for the worst case".

Make a model generalise well in any environment &lt;--&gt; generalise
well in the worst case.

Two methods:

<img src="media\image1.png" style="width:4.90403in;height:1.13531in" alt="图片" />

Formulation of adversarial environment 

For an adversarial environment, we have:

● **Adversarial goal** defines which environment is "worst" for your
model.

● **Strategy space** defines the space of possible environments from
which the devil can choose. (and they choose the worst from it)

● **Lower bound guarantee**: Your model's performance against the
worst-case environment provides a lower bound on its performance against
the space of possible environments.

● **Caveat**: The guarantee is only within the pre-set space of possible
environments. 

● **Knowledge** (Adversary's knowledge on the target model  -- what they
can get from the attacked model) defines the devil's ability to pick the
worst environment for your model. 

● **White box attack**: Access to model architecture and weights, or we
can obtain the input gradients from the model.

● **Black-box attack**: You observe only the inputs and outputs to a
model. 

White-box Attack

<img src="media\image2.png" style="width:4.88542in;height:1.66385in" alt="图片" />

1.FGSM (Fast Gradient Sign Method) attack

L-inf adversarial attack. 

<img src="media\image3.png" style="width:3.92694in;height:1.5463in" alt="图片" />

● **Goal**: Reduce classification accuracy, while being imperceptible to
humans.

● **Space**: For every sample, adversary may add a perturbation dx with
norm \|\|dx\|\|\_inf &lt; ε.

<img src="media\image4.png" style="width:2.00069in;height:1.18242in" alt="图片" />

● **Knowledge**: Access to model architecture and weights (white box
attack).

FGSM as L-inf attack

<img src="media\image5.png" style="width:2.38222in;height:0.30955in" alt="图片" />

where:

L: Loss value for model θ, input x, and ground-truth label y.

sgn: Takes value +1 for positive values, takes value -1 for negative
values.

ε: Size of the perturbation. It determines the L-inf norm of the
attack. 

We can show that \|\|ε·sgn(...)\|\|inf &lt; ε, so after the update, x is
still within the space.

2.PGD (Projected Gradient Descent) attack 

Lp adversarial attack (1 ≤ p ≤ ∞).

<img src="media\image6.png" style="width:3.03361in;height:1.14814in" alt="图片" />

● Goal: Reduce classification accuracy, while being imperceptible to
humans.

● Space: For every sample, adversary may add a perturbation dx with norm
\|\|dx\|\|p &lt; ε.

<img src="media\image7.png" style="width:2.04722in;height:1.20425in" alt="图片" />

● Knowledge: Access to model architecture and weights (white box
attack). 

PGD as Lp attack

<img src="media\image8.png" style="width:3.54542in;height:0.37523in" alt="图片" />

where:

t: Iteration index

α: Step size for each iteration.

<img src="media\image9.png" style="width:0.29778in;height:0.11167in" alt="图片" />:
Projection on the Lp sphere around x. 

<img src="media\image10.png" style="width:1.89833in;height:1.43402in" alt="图片" />

Short summary: FGSM vs PGD

● Optimisation problem for adversary is non-convex.

● No guarantee for the optimal solution, even within an epsilon ball
(it’s not tiny in a high-dim space).

● Strength of the attack depends a lot on the optimisation algorithm.

● PGD is generally much stronger than FGSM; PGD finds better optima.

● PGD is generally the state of the art attack even now. 

Exploring different strategy spaces 

(Use different ways to measure the change of original image.)

● So far: Pick perturbation inside an Lp ball (a norm 1 ball).

● Problem with Lp ball as strategy spaces: it is not aligned with
human's perception.

Small image translations result in small (huge) L2 distances but result
in perceptually huge (small) distances.

Example: Which pair looks more similar to each other? 

<img src="media\image11.png" style="width:3.90833in;height:1.17354in" alt="图片" />

左图：img and img after translation

<img src="media\image12.png" style="width:2.8475in;height:1.75519in" alt="图片" />

Can we define a metric that assigns small distances to small image
translations? --&gt; Optical flow! 

3.Flow-based attack

Optical flow distance

● Optical flow measures the smallest warping of the underlying image
mesh to transform x1 into x2 .

<img src="media\image13.png" style="width:4.07583in;height:1.14633in" alt="图片" />

● Optical flow is represented as a vector field over the 2D space.

<img src="media\image14.png" style="width:1.59125in;height:0.91194in" alt="图片" />

● Here, the size of the warping may be computed via total variation.

Adversarial flow-based perturbation

● Adversary warps the underlying mesh for the image according to f such
that the classification result is wrong. 

<img src="media\image15.png" style="width:4.81097in;height:2.06765in" alt="图片" />

● Strategy space: Flows f with the size of total variation less than δ.

● \|\| f \|\|\_{TV} ≤ δ

4.Physical attacks 

Lp attacks, optical flow attack and other attacks that alter the digital
image.

<img src="media\image16.png" style="width:5.5in;height:1.84974in" alt="图片" />

But do such adversaries exist in the real world? Sometimes basic
security technology can already prevent such adversaries.

So the more practical and more common attack is Physical attacks!

Physical attacks 

<img src="media\image17.png" style="width:3.19181in;height:1.86545in" alt="图片" />

Physically change the object in real world.

● Adversaries usually do have the necessary access to real-world
objects. 

<img src="media\image18.png" style="width:5.27625in;height:1.81025in" alt="图片" />

5.Other strategy spaces: Object poses in 3D world 

● Strategy space: Changes in object poses.

● Doesn't necessarily care about "small" changes in object pose. 

<img src="media\image19.png" style="width:4.11306in;height:1.23149in" alt="图片" />

Black-box attack

The white-box attack we discuss above requires input gradients from the
model, which is stronger but less realistic than black-box attack.

<img src="media\image20.png" style="width:4.69in;height:0.60069in" alt="图片" />

Many real-world applications are based on API access, so there are
further limitations:

○ Number of queries within a time window (rate limit).

○ Possible blocking of malicious query inputs.

Example:

<img src="media\image21.png" style="width:4.32708in;height:1.98311in" alt="图片" />

Black-box attack via substitute model 

**Step 1**: Gather input-output pairs from the original model (API
calls). <img src="media\image22.png" style="width:5.5in;height:0.68298in" alt="图片" />

**Step 2**: Train a substitute model g with the input-output pairs. g is
expected to be similar to the original model f.

<img src="media\image23.png" style="width:5.5in;height:0.64286in" alt="图片" />

Adversary uses g to generate white-box attacks.

Hope that this attack also works for f.

Usually it does attack f too.

Black-box attack via zeroth-order attack 

Attack in the same way as we attack white-box model, but we have to
approximate gradient

● One can indeed approximate model gradient only with API calls!

● For h small enough, one can approximate the gradient numerically:

<img src="media\image24.png" style="width:3.28486in;height:0.55002in" alt="图片" />

● But much less efficient than backpropagation.

Framework:

<img src="media\image25.png" style="width:3.75944in;height:1.63788in" alt="图片" />

The most challenging part in Algorithm above is to compute the best
coordinate update in step 3. Different methods exist.

Defense

There are many defense methods, but we'll touch upon one representative
defense method for the lecture.

Adversarial training 

Include attack samples in the training batch. 

<img src="media\image26.png" style="width:4.46667in;height:0.55962in" alt="图片" />

Instead of feeding samples from the distribution D directly into the
loss L, we allow the adversary to perturb the input first. This gives
rise to a saddle point problem (Minimax formulation).

One can interpret attack as a scheme for maximizing the inner part of
the saddle point formulation

● Caveat: Adversarial training is very difficult to perform at scale. 

Reasons why some other defense are not so effective? 

● Defenses are specifically targetted against gradient-based attacks.

● They only make the gradient malfunction - to mislead gradient-based
attacks.

● The model itself is still vulnerable.

● One can use slight modification of gradient-based attacks to attack it
again.

<img src="media\image27.png" style="width:3.49889in;height:2.1702in" alt="图片" />

Example: Input transformations 

By apply image transformations (and random combination of them)
(Cropping and rescaling, Bit-depth reduction, JPEG encoding +
decoding....), you can \*removing\* adversarial effects from the input
image.

<img src="media\image28.png" style="width:4.79236in;height:0.98495in" alt="图片" />

It's a successful defense for PGD, FGSM. But if we modify the attack by
incoperating the information of transformation, we could still attack it
successfully:

● Defense by cropping and rescaling (differentiable transformations):

○ Attack by differentiable transformation -- attack the joint network
that incoperate the transformation.

<img src="media\image29.png" style="width:4.07583in;height:1.49806in" alt="图片" /> 

● Defense by other discrete transformations:

○ Attack by differentiating "through" quantisation layers. (Use
straight-through estimator to estimate the gradient of discrete
transformations -- see it as identity mapping and ignore its gradient)

<img src="media\image30.png" style="width:4.31778in;height:1.55609in" alt="图片" />

● Defense by random mixture of transformations:

○ Attack by performing expectation over transformations (EOT):

<img src="media\image31.png" style="width:2.90333in;height:0.34765in" alt="图片" />

<img src="media\image32.png" style="width:4.46667in;height:1.58826in" alt="图片" />

(In practice, average all gradients (which all incoperate
transformation))

EoT can attack against gradient obfuscations, but there's a defense
against EoT -- BaRT

Barrage of Random Transforms (BaRT)

● Introduce 10 groups of possible image transformations.

> ○ Color Precision Reduction
>
> ○ JPEG Noise
>
> ○ Swirl
>
> ...
>
> ○ Denoising

● And apply them with random sequence.

<img src="media\image33.png" style="width:2.90333in;height:0.26663in" alt="图片" />

Certified defenses 

Some theoretical analysis (not so meaningful in practice) + deriving
defense method from it.

Given a two-layer neural network: f(x) = V σ(Wx) (assumption too strong)

, where V and W are matrices, and σ( ) is a non-linearity with bounded
gradients. E.g. ReLU or sigmoid. 

We write the following for the worst-case adversarial attack. 

<img src="media\image34.png" style="width:2.70792in;height:0.33757in" alt="图片" />

("A" means attack)

They have produced the following upper bounds on the severity of
adversarial attack:

<img src="media\image35.png" style="width:4.94125in;height:0.38878in" alt="图片" />

,
where<img src="media\image36.png" style="width:1.27486in;height:0.13028in" alt="图片" />a
term

To defense, we can incoperate the second term in loss in some way, and
formulate the following optimization problem instead of ERM:

<img src="media\image37.png" style="width:5.5in;height:0.37167in" alt="图片" />

Given V\[t\], W\[t\] and c\[t\] values at iteration t, one obtains the
guarantee for any attack A:

<img src="media\image38.png" style="width:4.74583in;height:0.36685in" alt="图片" />

(The severity of attack is reduced)

History

<img src="media\image39.png" style="width:4.71792in;height:2.27991in" alt="图片" />

<img src="media\image40.png" style="width:5.00639in;height:2.09694in" alt="图片" />

2020 - : Stop the cat and mouse game!

It's a dead end.

Future

Considered attacks are way too strong. Instead more work towards less
pessimistic defense:

● Defense against black box attacks.

● Defense against non-adversarial, non-worst case perturbations:

● OOD generalisation

● Domain generalisation

● Cross-bias generalisation. 

Alternatives:

1\. Diversify and randomise (eg BaRT)

2\. Certified defenses

3\. Deal with realistic threats, rather than unrealistic worst-case
threats. 

Summary

● DNNs are vulnerable within small Lp ball.

● Attacks and defenses tend to be a cat and mouse game.

● People seek alternative directions, such as domain generalisation
