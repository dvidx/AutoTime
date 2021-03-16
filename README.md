# AutoTime
Automated Simple Times Series Module

(Work in Progress)

This is a simple tool to create a forecasts for any time series data.

Usage:
- Adjust config to your needs
- Execute _main_.py

What does this tool do?
- Automated Feature Generation
- Grid Search to find optimal model parameters and time lag
- Create a variation of visualization that help analyse the forecast performance

Requirements:
- Data needs to be pre-processed (Ensure it is continous)
- Config needs to be adjusted according to needs

In general more and data of higher quality will yield better results.
The opposite is true for bad data -> Garbage in, Garbage out.
Deep Learning models only process numerical data, therefore everyting else will be dropped.

Config:
- In- and output locations
- Forecast Horizon
- Train samples (rest will be test samples)
- Grid search parameter search space


Future Development:
- Integrate the generation of visualization into python instead of R
- Develop more Time Series models and allow to choose multiple
- More options for Feature Engineering
