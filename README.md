# moe-echem-additives
Mixture-of-Experts for electrolyte additive discovery in Zn–I flow batteries: deep data mining, interpretable MoE architectures, and reproducible pipelines.
**Authors:** Xianglin Wang<sup>a</sup>, Qicheng Chen<sup>a*</sup>, Nan He<sup>a</sup>, Dong Wang<sup>b</sup>, Binjian Nie<sup>c</sup>, Yingjin Zhang<sup>d</sup>, Wei Liu<sup>d</sup>  
(*Corresponding author: Qicheng Chen*)

> This repository accompanies the paper **“Mixture of Experts-Driven Deep Data Mining for Electrolyte Additive Design in Zinc–Iodine Flow Batteries.”**  
> It provides code, data schemas, and reproducible pipelines for expert-driven multimodal mining and interpretable screening of electrolyte additives in Zn–I flow batteries (ZIFBs).

---

## Overview

We present a **Mixture-of-Experts (MoE)** framework that unifies:
- **Deep data mining** across textual literature, structural chemistry, and experiment logs;
- **Interpretable expert heads** tailored to complementary signals (e.g., molecular descriptors, interaction energetics, prior knowledge);
- **Routing & gating** to adaptively weigh experts for **additive ranking**, **property prediction** (e.g., CE/EE proxies), and **design insights**.

The pipeline aims to accelerate **electrolyte additive discovery** for **Zinc–Iodine Flow Batteries (ZIFBs)** with transparent modeling and reproducibility.

---

## Key Contributions

- **MoE architecture for materials discovery:** Specialized experts (e.g., descriptor expert, graph expert, text/literature expert) with a data-driven router.
- **Interpretable routing:** Per-sample expert attributions to understand *why* an additive is recommended.
- **Reproducible mining pipeline:** From raw sources (papers, databases) to curated features and trained models.
- **Task bundle:** Additive ranking, target property prediction (e.g., surface free energy proxies), and ablation tooling.

---

## Repository Structure
