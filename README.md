# Simple Empirical Model for Predictions of UV Luminosity Functions at High Redshift

The method is described in [Shen et al. 2024](https://arxiv.org/abs/2406.15548).

## `model_uvlf.ipynb`
The main pipeline to generate predictions of the UV luminosity functions using the empirical model described in Shen et al. 2023,2024. 

## `utilities/*`
Code for empirical galaxy formation models and UVLF processing. Make sure you have **hmf**, **camb**, and **astropy** installed.

## `sampling.ipynb`
A Monte-Carlo sampling approach to estimate the combined variability from different physical sources

## compiled UVLF observation data
The compiled observational data is stored at `./observational_data/`. The data format for each observation paper could be slightly different, please check the header of the data file carefully.
