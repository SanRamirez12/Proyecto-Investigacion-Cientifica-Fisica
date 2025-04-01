from fermipy.gtanalysis import GTAnalysis

gta = GTAnalysis('config.yaml')

gta.setup()

gta.write_roi('fit0')

gta.optimize()

gta.write_roi('fit1')

gta.free_sources(distance=1.0,pars='norm')

gta.fit()

gta.sed('mkn421')

gta.write_roi('fit2')
