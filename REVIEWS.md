# CulinaryCut — Reviewer Feedback (3 reviewers)

Logged on the rebuttal cycle (Apr-May 2026).

---

## Reviewer 1Tj — Borderline Accept (4) · Confidence 2/Low

**Summary**: VLA benchmark with deformation, topology change, varying forces.
Two task families (spatial-based, division-based multi-cut). 5k trajectories,
≥5 augmented language descriptions each. Three baselines (OpenVLA, Octo, RDT-1B);
biggest gap on multi-cut (long-term geometric planning).

### Strengths
- Convincing problem framing.
- Technically sound dataset creation, literature-grounded properties, real-world validation on banana.
- Extensive evaluation (multi-object, cross-object transfer, multi-cut).
- Clear writing, good limitations section.

### Major Weaknesses
- **Fine-tuning setup not described clearly.** Need details to ensure performance gaps reflect model limitations, not training shortfall.

### Minor
- Table 4: two-line heading or larger spacing for readability.

### Justification
> CulinaryCut is a well-motivated benchmark that addresses a real gap... The
> only minor concern is the fine-tuning procedure of baseline models, that
> requires a bit more details... I am open to increase rating after rebuttal.

### Rebuttal Asks
- Comment on baselines' fine-tuning procedure.

---

## Reviewer gYxG — Borderline Reject (3) · Confidence 3/Moderate · **Continuity**

**Summary**: Food cutting for VLAs. CulinaryCut benchmark, ManiSkill 7-DoF arm
with knife EE, MLS-MPM deformable, human teleop + motion planner. Template
language. Tasks: randomized poses, multi-object, unseen-object generalization,
continuous-cut, split-cut. Evaluates 3 SoTA VLAs. Identifies two limitations:
(i) no notion of physical properties, (ii) struggles with object size / cut
ratios. Proposes simple safety module → motivates PVLA.

### Strengths
- Showcases VLA limitations on complex tasks.
- Real-world force validation.
- Large multi-modal dataset (demos, language, force).
- Small real-robot evaluation included.

### Major Weaknesses
1. **Qualitative material too limited.** Single-banana clips; real-world clip
   incomplete (no clear pull-out). PowerPoint unpolished — physics-aware action
   *less* successful than normal in orange clip; "RDT-1B vs Failure Case" row
   shows three failures. Diversity claims need scale qualitative results.
2. **Confused terminology.** Pretrained VLAs + safety module as velocity
   guardrail, but policy receives no force info (L277). Calling this
   "force-aware control" (supplementary E.4) is misleading vs ForceVLA-style
   force-aware policies.
3. **PVLA motivation contradicts setup.** Teaser figure + safety module
   description suggests authors already use "physics-aware" VLAs, yet the paper
   motivates PVLA as future work. Define PVLA clearly. Provide quantitative
   experiments **without** the safety module — do baselines fail entirely
   without the velocity guardrail?
4. **Story flow misaligned.** Intro identifies physics + geometry gap; §3.2 says
   tasks expose geometry gap and *sequential planning* gap.
5. **Camera setup unclear.** Multi-view rendered from a single position (B.7) —
   how is position/orientation varied? Don't different views create ambiguity
   for instructions like "cut from the left side"?
6. **Cutting-style coverage missing.** Several cutting styles described in
   supplementary — does main paper evaluation use randomized styles, or only
   normal cut? Either way, need style-stratified benchmark numbers.
7. **Real-world success criterion strange.** "Two-thirds of cross section" vs
   the obvious "cut all the way through". Why this limited definition?
   Where is this metric actually used quantitatively?
8. **Missing baseline-execution details.**
   - VLA update rate?
   - How many future actions predicted?
   - Trained on randomized views but inference uses only robot perspective?
9. **Promised-but-not-delivered carry-overs from prior cycle:**
   - 7-object set still small.
   - Inconsistent task numbering.
   - Fig. 4 legend vs bar color mismatch.
   - Direction frame ambiguity ("right side" robot- or camera-frame?) — Fig. 8b.
   - Evaluation N=20 → too small, 0/100% extremes.
   - More qualitative examples for scale shifts and clutter.

### Minor
- Fig. 4 / Fig. 7 redundant with Tab. 4.
- Tab. 4: vertical separators (esp. model row).
- L372: "as shown in the main table" — be specific.
- Fig. 6: what does "average" mean? Average cut ratio (=0.5)?

### Justification
> Viable direction, potential to be useful benchmark. Main concerns: low
> quality/quantity of qualitative results, ambiguities around VLA vs PVLA and
> force-awareness, missing protocols on task setup.

### Rebuttal Asks
- Clarify experimental setup.
- Resolve VLA / PVLA / "force-aware" terminology ambiguity.
- Address remaining carry-over items if space permits.

---

## Reviewer vsDG — Borderline Accept (4) · Confidence 3/Moderate

**Summary**: New dataset for robotic food cutting. Addresses limitations of
existing rigid-object datasets. Physics-based simulator validated on real arm.
5k trajectories with multiple language instructions per cut.

### Strengths
- Demonstrates motion-only approaches insufficient for deformable + topological
  food cutting.
- Combined physics + robot simulator, 325K trajectories across 7 food categories.
- Most comprehensive food-cutting dataset to date.
- Real-arm validation.

### Major Weaknesses
1. **Missing surgical robotics literature.** Surgical robotics community works
   on cutting deformable soft tissue — same challenges. No references / discussion
   (cites IEEE 10802347 as example). What can surgical robotics offer here?
2. **Segmentation sim-to-real gap.** Real scenes have touching / occluded
   objects on a plate; the simulator assumes perfect segmentation. Quality of
   segmentation directly impacts grounding. Paper claims "visual realism does not
   guarantee physical consistency" — reviewer counters: physical consistency
   *depends* on accurate segmentation. Address this form of sim-to-real gap.
3. **No object-securing modeled.** Cut = single downward pass with no
   stabilizing hand. Humans secure with one hand and cut with the other; without
   securing the object, some geometries should jump or slide. Do you model
   surface friction to predict this? Is this a realistic cutting setting?

### Minor
- None.

### Justification
> Well-motivated, comprehensive dataset, real-robot validated. Niche
> application; surprised surgical robotics not cited. Valuable to food-cutting
> robotics researchers.

### Rebuttal Asks
- Address the three weakness points.

---

# Cross-Reviewer Synthesis

## Recommendation Spread
| Reviewer | Score | Confidence | Stance |
|---|---|---|---|
| 1Tj  | 4 (BA) | 2 (Low)  | Easy lift to Accept; just clarify fine-tuning |
| gYxG | 3 (BR) | 3 (Mod)  | **Hardest reviewer**, continuity from previous cycle |
| vsDG | 4 (BA) | 3 (Mod)  | Lift to Accept if surgical / segmentation / securing addressed |

**The decisive reviewer is gYxG.** They are: (a) the only Borderline Reject,
(b) explicitly continuing from a previous cycle, (c) noting that several promised
fixes from the prior cycle were *not* delivered. ACs weigh "didn't address prior
feedback" heavily — every carry-over item must be visibly fixed.

## Themes Across Reviewers (where ≥2 reviewers complain)

**Almost no overlap.** Each reviewer hits a different flank:
- **Methodology rigor / details** (1Tj fine-tuning, gYxG baseline-execution details)
  — both ask "is the comparison fair?" → unify the answer.
- **Realism of the cutting setting** (vsDG securing/friction, gYxG segmentation
  is implied real-world; gYxG real-world success criterion)
  — touches the sim-to-real gap.

## Carry-Over Items (gYxG explicitly flagged as "promised, not delivered")

These are the **highest-leverage** items because they signal "authors don't
respond to feedback":

| # | Item | Effort |
|---|---|---|
| C1 | Expand object set beyond 7 | **In progress** — 15 fruits already in `configs/fruits/`, 14 with real Kenney CC0 meshes |
| C2 | Consistent task numbering throughout | Easy |
| C3 | Fig. 4 legend/bar color match | Easy |
| C4 | Direction frame: robot vs camera-perspective | Clarify in caption + Sec.3 |
| C5 | Evaluation N≥? (mitigate 0/100% extremes) | Need re-runs — costly |
| C6 | More qualitative examples (scale shifts, clutter) | Need re-renders + curation |

## Severity-Ordered Action List

### 🔴 Critical (blocking for gYxG → Accept)
1. **Resolve VLA vs PVLA terminology.** Define PVLA. State explicitly whether
   `VLA + safety module = PVLA`. If yes, run the **without-safety-module
   ablation** for all baselines and report. (gYxG #2,3)
2. **Carry-over items C1-C6** must visibly land in the rebuttal PDF / supplementary.
   The 14 real fruit meshes work currently in progress directly addresses C1.
3. **Qualitative results at scale.** Curated supplementary site/video with:
   diverse fruit cuts, multi-object scenes, clutter, scale shifts, success +
   failure cases per model. PowerPoint should be replaced or polished.
   (gYxG #1)

### 🟠 Major (lifts 1Tj and vsDG)
4. **Fine-tuning details.** Per-baseline learning rate, optimizer, batch,
   epochs, LoRA rank if any, total compute, train/val split, when did each model
   converge. (1Tj major; gYxG #8)
5. **Camera setup.** Diagram showing single-position multi-view rendering
   logic. Resolve "left side" ambiguity in language. Specify training-vs-inference
   view discrepancy. (gYxG #5; vsDG #2 partially)
6. **Cutting-style stratification.** Either label main results as
   "normal-cut only" or run benchmark across the cutting styles described in
   supp. and add Tab. (gYxG #6)
7. **Real-world success criterion.** Either justify two-thirds threshold
   (probably: knife-stop position is at 67% blade depth) or replace with full
   cut-through and report results. Explain where the metric is used. (gYxG #7)

### 🟡 Important (closes vsDG cleanly)
8. **Surgical robotics literature.** 1-2 paragraphs in related work; cite the
   reviewer-suggested IEEE doc + 3-5 others (deformable tissue cutting,
   needle insertion mechanics, FEM-based cutting). (vsDG #1)
9. **Object-securing model.** Either:
   - (a) Add second gripper/fixture in simulator + redo a small subset to show
     it stabilizes, OR
   - (b) Quantify object motion during cut (centroid drift) and discuss when
     securing would be necessary. (vsDG #3)
10. **Segmentation sim-to-real.** Plot showing how perfect-mask vs noisy/
    instance-segmentation affects grounding accuracy. Discuss as future work.
    (vsDG #2)

### 🟢 Cleanup (cheap polish)
11. Tab. 4 layout — vertical separators + two-line headers. (1Tj minor; gYxG minor)
12. Drop redundant Fig. 4 / Fig. 7 bar plots. (gYxG minor)
13. L372 "main table" → specific reference. (gYxG minor)
14. Fig. 6 caption — define "average". (gYxG minor)
15. Story flow — align intro (physics + geometry gap) with §3.2 (geometry +
    sequential planning gap). Pick one taxonomy. (gYxG #4)

## Quick-Win Calculation
If we ship: **#1, #4, #5, #11-15 + carry-overs C1-C6** in the rebuttal
PDF/supplementary, gYxG plausibly moves 3 → 4, 1Tj to 5, vsDG to 5. That's
**3× Accept = clear accept**.

If we *also* ship: **#3 (curated qualitative material)** and **#8 (surgical
robotics paragraph)**, gYxG moves to 5 → very strong.

#9 and #10 are "nice but not necessary" — list as future work in the rebuttal
if time-constrained.

## Current Repo Status (as of session)
- Multi-fruit (15) infrastructure in place — **C1 partially addressed**.
- 4-GPU sweep currently running with new real Kenney CC0 meshes. Output will
  feed: (a) qualitative variety (#3), (b) per-fruit force/velocity tables for
  expanded experiments.
- Material variation experiment script now reports **raw simulator forces**
  (DISPLAY_SCALE post-hoc fudge removed) — defensible if reviewers ask about
  validation magnitudes.
- Velocity variation script logs **commanded vs actual blade speed** — useful
  if asked about kinematic consistency / resistance modeling.
