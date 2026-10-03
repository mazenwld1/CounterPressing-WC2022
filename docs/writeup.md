# Detecting Counterpressing in the FIFA World Cup 2022 from StatsBomb Event and 360 Data

*Adapting the supervised approach of Bauer & Anzer (2021) to publicly available data*

---

## Abstract

Counterpressing, the immediate attempt to win the ball back after losing it, is usually identified by analysts watching video. Bauer and Anzer (2021) showed that it can be detected automatically with an XGBoost model trained on synchronized tracking and event data and expert labels, reaching an AUC of 0.874. This project asks how much of that result can be reproduced with public data: StatsBomb event data and 360 freeze-frames for all 64 matches of the 2022 FIFA World Cup. Because the public data has no continuous tracking and no expert labels, I used a single freeze-frame at the moment of each turnover to build features, and used StatsBomb's own `counterpress` flag as a proxy label. Evaluated with nested, repeated cross-validation that holds out whole matches, XGBoost reached a ROC-AUC of 0.684 on all turnovers, against 0.600 for the paper's 5-yard rule-based baseline. A large share of this advantage came from turnovers won by the goalkeeper, which are almost never counterpressed (3.4% vs. 56.5% for other turnovers). When these were removed, XGBoost reached 0.597 against 0.547 for the baseline: a consistent but modest improvement. The main predictive signal is the proximity of the losing team to the ball and the distance to the sideline, which agrees with the paper's findings and with practitioner rules of thumb. A second target, regaining the ball within five seconds, was only weakly predictable (ROC-AUC 0.630).

---

## 1. Introduction

After a team loses the ball, it can either press immediately to win it back (counterpressing) or fall back into a defensive shape. The first seconds after a turnover are important because the team that has just attacked is usually badly positioned to defend. Bauer and Anzer (2021) automated the identification of counterpressing with a supervised model. They used positional (tracking) and event data from the German Bundesliga, 134 expert-designed features, and more than 11,000 turnovers hand-labelled by trained student analysts.

That data is proprietary. The goal of this project is to build a comparable pipeline on open data and to be honest about what is lost. The research questions are:

1. How well can counterpressing be detected from a single 360 freeze-frame plus event context?
2. Does a machine learning model outperform the simple rule-based baseline used in the paper?
3. Which factors drive the predictions, and do they agree with the paper's findings?
4. Can the same features predict whether the ball is regained within five seconds?

## 2. Data

I used the StatsBomb open data for the 2022 FIFA World Cup (64 matches): event data, which logs ball-related actions, and 360 data, which gives the location of the visible players at the moment of each event. Pitch coordinates are in StatsBomb yards (120 × 80).

Two properties of this data differ from the paper and shape every design choice below:

- **360 frames are a single instant with partial visibility.** Only players in the camera's visible area are recorded, and there is no information about movement in the seconds after the turnover. The paper's strongest features (team speeds in the first two seconds) cannot be reproduced.
- **There are no expert labels.** The only available indicator of counterpressing is StatsBomb's own `counterpress` flag, which is produced by the provider's rules.

## 3. Method

### 3.1 Turnover detection

Turnovers were taken from consecutive StatsBomb possessions that belong to different teams. Following the paper's exclusion criteria, I kept only possessions that start in open play (`Regular Play` or `From Counter`) and only turnovers where the ball was lost in the losing team's attacking half. This gave 3,155 turnovers. Of these, 2,823 (89.5%) had a usable 360 freeze-frame; the rest were dropped so the model and the baseline are compared on identical data. In the remainder, the team that lost the ball is called team A (the potential counterpressing team) and the team that won it is team B.

### 3.2 Labels

- **`y_cp` (counterpressing):** 1 if team A has an event flagged `counterpress` within five seconds of the turnover. This is a proxy: the model learns to reproduce the provider's rule, not an analyst's judgement.
- **`y_regain` (success):** 1 if team A performs an on-ball action (ball recovery, interception, pass, carry, dribble or shot) within five seconds, which corresponds to the paper's five-second success definition. I did not use a change in StatsBomb's possession identifier for this, because inspecting the raw events showed that possessions persist through contested duels and quick regains, so a possession change under-counts them.

### 3.3 Features

All features describe the situation at the instant of the turnover, from team A's perspective, with coordinates oriented so that team A attacks in the same direction. Nothing computed from events after the turnover is used, because the labels are derived from those events and they would leak the answer. Goalkeepers are excluded from the freeze-frame features, as in the paper.

- **Location and context:** turnover position, distance to the opponent's goal, distance to the nearest sideline, period and minute, the type of event that lost the ball, whether it was under pressure, and the duration, length and passing count of the previous possession.
- **Local numbers around the ball:** the number of players of each team within 5, 10, 20 and 30 yards, the numerical superiority within 10 yards, and the distance of the nearest team-A player.
- **Shape (adapted from the paper's Table 1):** the number of players in front of and behind the ball, local stretch indices of the closest 3, 4 and 5 team-A players, convex-hull area, team centre, and spread in both dimensions.
- **Visibility:** the number of visible players per team, which controls for camera coverage.

The event type with which team B first wins the ball was excluded after it proved to be a near-perfect shortcut (98% of turnovers won with a carry were labelled as counterpressing in one run). I believe it reflects how a carry is logged, not tactics.

### 3.4 Models and baseline

I used XGBoost, with hyperparameters selected by Bayesian optimization (a tree-structured Parzen estimator via Optuna), as in the paper. A second variant added a class-balancing weight. The baseline is the paper's naive rule: a turnover counts as counterpressed if a team-A player is within five yards of the ball.

### 3.5 Evaluation

Turnovers from the same match are strongly correlated, so a random split would leak information between training and test data. I used nested, repeated, grouped cross-validation: whole matches are assigned to five folds, the assignment is repeated three times with different shuffles, and hyperparameters are tuned only on the training matches of each fold (15 evaluations in total). I report ROC-AUC and PR-AUC, which do not depend on a threshold, the Matthews correlation coefficient (MCC) and balanced accuracy at a 0.5 threshold, and lift in the top decile of predictions. I do not report F1: with roughly 50% positives, predicting "counterpress" for everything already scores an F1 near 0.67, so it makes weak models look good. Values are mean (standard deviation across folds). The folds overlap across repeats, so the standard deviations understate the true uncertainty, and differences should be described as consistent, not as statistically significant.

## 4. Results

### 4.1 Detecting counterpressing

Of the 2,823 turnovers with a frame, 326 (11.5%) were won by the goalkeeper. Only 3.4% of these were counterpressed, against 56.5% of the remaining 2,497 turnovers, and none of them was regained within five seconds. They are a trivial case, which is also noted in the paper, where most "fallback" situations began with a goalkeeper catching the ball. I therefore report the results with and without them.

| Setting | Model | ROC-AUC | PR-AUC | MCC | Balanced acc. | Lift@10% |
|---|---|---|---|---|---|---|
| All turnovers (base rate 0.504) | Naive 5-yard rule | 0.600 (0.011) | 0.562 | 0.214 | 0.600 | n/a |
| | XGBoost | 0.684 (0.017) | 0.639 | 0.299 | 0.639 | 1.35 |
| | XGBoost + class weight | 0.684 (0.017) | 0.638 | 0.296 | 0.639 | 1.33 |
| **Without goalkeeper wins (base rate 0.566)** | Naive 5-yard rule | 0.547 (0.015) | 0.590 | 0.107 | 0.547 | n/a |
| | **XGBoost** | **0.597 (0.017)** | **0.638** | **0.149** | **0.560** | **1.17** |
| | XGBoost + class weight | 0.597 (0.018) | 0.638 | 0.134 | 0.567 | 1.19 |

*Without goalkeeper wins, the raw pitch coordinate `y` was also dropped, because it is redundant with the distance to the sideline and its direction of effect was not interpretable.*

The naive rule reached an AUC of 0.600 on all turnovers, which is almost identical to the 0.602 reported for the same rule in the paper, and supports the comparability of the baseline. XGBoost's advantage over it is 0.084 AUC on all turnovers and 0.050 on the stricter set. The class-weighted variant gave no benefit.

### 4.2 Predicting a regain within five seconds

| Model | ROC-AUC | PR-AUC | Lift@10% |
|---|---|---|---|
| Naive 5-yard rule | 0.582 (0.029) | 0.077 | n/a |
| XGBoost | 0.615 (0.031) | 0.112 | 1.61 |
| XGBoost + class weight | 0.630 (0.029) | 0.116 | 1.82 |

The base rate is 0.064 (about 180 positive turnovers). With such a rare class, MCC is uninformative, and the unweighted model predicts "no regain" for everything at a 0.5 threshold. The ranking quality is only modestly better than the baseline. Since this run included goalkeeper-won turnovers, none of which were regained, these figures are likely to be optimistic.

### 4.3 What drives the predictions

SHAP values of a model fitted on all matches (used for interpretation only, never for performance) show:

- **Proximity of team A to the ball** is the strongest factor. When no team-A player is close to the ball, the predicted probability of counterpressing drops sharply. This is partly built into the label, since StatsBomb's flag depends on pressure events, but it agrees with the paper's emphasis on players close to the ball.
- **Distance to the sideline:** turnovers near the sideline make counterpressing more likely. This supports one of the paper's rules of thumb (RQ3).
- **Players ahead of the ball:** a larger number of team-A players in front of the ball raises the probability, in line with the idea that counterpressing means defending up the pitch instead of retreating.
- In the version that included the goalkeeper turnovers, two features were dominant that disappear from the top of the ranking once those turnovers are removed: the distance to the opponent's goal and the number of team-B players within five yards of the ball. They mostly identified goalkeeper situations.
- Features that depend on how many players the camera captured (visible-player counts, hull area, spread) appear in the plots, but I do not interpret them tactically.

## 5. Discussion

Compared with the paper, the achievable performance on public data is much lower (an AUC of about 0.60-0.68 against 0.874). This is expected given two differences. First, the features are limited to a single snapshot and cannot capture how players move in the first two seconds, which were among the paper's most influential features. Second, the label is a provider's rule instead of an expert annotation, so the model's ceiling is determined by how predictable that rule is from the snapshot. A proximity-based rule applied to a snapshot recovers part of it, but not all.

The comparison without goalkeeper wins is the more honest estimate of performance on real open-play transitions. In that setting the model is consistently better than the five-yard rule, but only modestly: the top 10% of ranked turnovers are 17% more likely to be counterpressed than average. The gap between the two settings is also a useful finding in itself. It shows that the headline AUC on all turnovers rewards a model for recognizing a situation (the goalkeeper has the ball) and not for recognizing a tactic, and that the paper's remark that the situation largely dictates the response applies here as well.

## 6. Limitations and future work

- **Proxy labels.** The target is a provider's rule, so the results measure agreement with StatsBomb's tagging, not with an analyst's definition. Hand-labelling a small sample of turnovers and measuring agreement with the flag would quantify this.
- **Single snapshot, partial visibility.** Movement after the turnover and players outside the camera view are unavailable.
- **Small sample.** 64 matches, and only about 180 regains, give wide uncertainty. Results from a single tournament may not generalize to league football.
- **Undefined situations.** The paper's experts discarded around 62% of turnovers as having no identifiable strategy (for example long balls after goal kicks). I filtered goalkeeper wins, but other such cases remain in the data, which probably lowers every score.
- **Regain target.** The stricter filtering was not applied to the regain results, and the label counts any on-ball action by the losing team, which can include actions that do not represent real control.
- **Future work.** Compare counterpressing with fall-back decisions, use expected-goals or shot-based outcomes as in the paper, add features from the preceding events to approximate movement, and evaluate on a larger competition.

## References

- Bauer, P., & Anzer, G. (2021). Data-driven detection of counterpressing in professional football: A supervised machine learning task based on synchronized positional and event data with expert-based feature extraction. *Data Mining and Knowledge Discovery, 35*, 2009-2049. https://doi.org/10.1007/s10618-021-00763-7
- StatsBomb. Open Data (event data and 360 data), FIFA World Cup 2022. https://github.com/statsbomb/open-data
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785-794.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30*.
