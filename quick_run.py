import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano.tensor as tt
warnings.simplefilter(action="ignore", category=FutureWarning)
az.style.use("arviz-darkgrid")
print(f"Running on PyMC3 v{pm.__version__}")
print(f"Running on ArviZ v{az.__version__}")
#
# with pm.Model() as model:
#     # Model definition
#     pass
#
# with pm.Model() as model:
#     mu = pm.Normal("mu", mu=0, sigma=1)
#     obs = pm.Normal("obs", mu=mu, sigma=1, observed=np.random.randn(100))
#
# print(model.basic_RVs)
#
# print(model.free_RVs)
#
# print(model.observed_RVs)
#
# print(model.logp({"mu": 0}))
#
# model.logp({mu: 0.1})
# logp = model.logp
# logp({mu: 0.1})
#
# print(dir(pm.distributions.mixture))
#
# with pm.Model():
#     x = pm.Normal("x", mu=0, sigma=1)
#
# x.logp({"x": 0})
#
# with pm.Model():
#     obs = pm.Normal("x", mu=0, sigma=1, observed=np.random.randn(100))
#
# with pm.Model():
#     x = pm.Normal("x", mu=0, sigma=1)
#     y = pm.Gamma("y", alpha=1, beta=1)
#     plus_2 = x + 2
#     summed = x + y
#     squared = x ** 2
#     sined = pm.math.sin(x)
#
# with pm.Model():
#     x = pm.Normal("x", mu=0, sigma=1)
#     plus_2 = pm.Deterministic("x plus 2", x + 2)
#
# with pm.Model() as model:
#     x = pm.Uniform("x", lower=0, upper=1)
#
# with pm.Model() as model:
#     x = pm.Uniform("x", lower=0, upper=1, transform=None)
#
# print(model.free_RVs)
#
import pymc3.distributions.transforms as tr
#
# with pm.Model() as model:
#     # use the default log transformation
#     x1 = pm.Gamma("x1", alpha=1, beta=1)
#     # specify a different transformation
#     x2 = pm.Gamma("x2", alpha=1, beta=1, transform=tr.log_exp_m1)
#
# print("The default transformation of x1 is: " + x1.transformation.name)
# print("The user specified transformation of x2 is: " + x2.transformation.name)
#
# class Exp(tr.ElemwiseTransform):
#     name = "exp"
#
#     def backward(self, x):
#         return tt.log(x)
#
#     def forward(self, x):
#         return tt.exp(x)
#
#     def jacobian_det(self, x):
#         return -tt.log(x)
#
#
# with pm.Model() as model:
#     x1 = pm.Normal("x1", 0.0, 1.0, transform=Exp())
#     x2 = pm.Lognormal("x2", 0.0, 1.0)
#
# lognorm1 = model.named_vars["x1_exp__"]
# lognorm2 = model.named_vars["x2"]
#
# _, ax = plt.subplots(1, 1, figsize=(5, 3))
# x = np.linspace(0.0, 10.0, 100)
# ax.plot(
#     x,
#     np.exp(lognorm1.distribution.logp(x).eval()),
#     "--",
#     alpha=0.5,
#     label="log(y) ~ Normal(0, 1)",
# )
# ax.plot(
#     x,
#     np.exp(lognorm2.distribution.logp(x).eval()),
#     alpha=0.5,
#     label="y ~ Lognormal(0, 1)",
# )
# plt.legend()
# plt.show();
#
# Order = tr.Ordered()
# Logodd = tr.LogOdds()
# chain_tran = tr.Chain([Logodd, Order])
#
# if __name__ == "__main__":
#     with pm.Model() as model:
#         mu = pm.Normal("mu", mu=0, sigma=1)
#         sd = pm.HalfNormal("sd", sigma=1)
#         obs = pm.Normal("obs", mu=mu, sigma=sd, observed=np.random.randn(100))
#
#         step1 = pm.Metropolis(vars=[mu])
#         step2 = pm.Slice(vars=[sd])
#         idata = pm.sample(10000, step=[step1, step2], cores=4, return_inferencedata=True)
#
#         az.plot_trace(idata)
#         plt.show;
#         print(az.summary(idata));
#         az.plot_forest(idata, r_hat=True)
#         plt.show();
#         az.plot_posterior(idata)
#         plt.show();
#         with pm.Model() as model:
#             x = pm.Normal("x", mu=0, sigma=1, shape=100)
#             idata = pm.sample(cores=4, return_inferencedata=True)
#
#         az.plot_energy(idata)
#         plt.show();
#
#         with model:
#             post_pred = pm.sample_posterior_predictive(idata.posterior)
#         # add posterior predictive to the InferenceData
#         az.concat(idata, az.from_pymc3(posterior_predictive=post_pred), inplace=True)
#         fig, ax = plt.subplots()
#         az.plot_ppc(idata, ax=ax)
#         ax.axvline(data.mean(), ls="--", color="r", label="True mean")
#         ax.legend(fontsize=10)
#         fig.show();
    # with pm.Model() as m0:
    #     x = pm.Uniform("x", 0.0, 1.0, shape=2, transform=chain_tran, testval=[0.1, 0.9])
    #     trace = pm.sample(5000, tune=1000, progressbar=False, return_inferencedata=False)
    #
    # _, ax = plt.subplots(1, 2, figsize=(10, 5))
    # for ivar, varname in enumerate(trace.varnames):
    #     ax[ivar].scatter(trace[varname][:, 0], trace[varname][:, 1], alpha=0.01)
    #     ax[ivar].set_xlabel(varname + "[0]")
    #     ax[ivar].set_ylabel(varname + "[1]")
    #     ax[ivar].set_title(varname)
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    data = np.random.randn(100)
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=1)
        sd = pm.HalfNormal("sd", sigma=1)
        obs = pm.Normal("obs", mu=mu, sigma=sd, observed=data)

        idata = pm.sample(return_inferencedata=True)

    with model:
        post_pred = pm.sample_posterior_predictive(idata.posterior)
    # add posterior predictive to the InferenceData
    az.concat(idata, az.from_pymc3(posterior_predictive=post_pred), inplace=True)

    fig, ax = plt.subplots()
    az.plot_ppc(idata, ax=ax)
    ax.axvline(data.mean(), ls="--", color="r", label="True mean")
    ax.legend(fontsize=10)
    plt.show()