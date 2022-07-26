# Quantum Supply Chain Manager

The quantum supply chain manager is a quantum solution for logistics problems. We use the power of quantum machine learning for product [backorder](https://www.investopedia.com/terms/b/backorder.asp) prediction and quantum optimization for finding the best route to pick those products with high demand to store them in strategic warehouses. With the first technique, our clients can be prepared for increasing the production of their products when they are in high demand. Once we have established a set of products needed during a period of time, we use our second solution, the vehicle routing problem (VRP) solution to find the optimal route for picking these products. This reduces considerably costs associated with logistics, transportation, backorders, and overstocking for our clients. In summary, these solutions will improve business in terms of client satisfaction, backorder and shipping transportation costs.

<img src="./Images/QSCM.png" width=900>

### But, why this is important?

Mainly, backorders indicate that the demand for a specific product was not well understood at a certain point in time and needs to be reevaluated. This has a high impact on the customers' loyalty and the company revenue. The solution to this problem must be increasing the production of the products with a high probability of backorder before that happen avoiding overstocking them which may result in high inventory costs. Using quantum neural networks (QNN), we can improve the predictability, the training size, and the number of variables for the backorder predictions.

On the other hand, the VRP finds optimal solutions for picking products in terms of costs and time. The benefit of finding an optimal solution can represent a saving of up to 30% of the costs associated with transportation [1]. However, current techniques deal with up to 400 variables in a reasonable time. Soon with a growing e-market, the capabilities of such models will be overcome. In the near future, we will need systems able to solve thousands of variables in small periods of time. 

### Is this something that we cannot do with classical computation?

The short answer is "depends", we can rely on the power of classical computation and heuristic methods to solve such a problem at **small scale**. At this moment, classical algorithms have the capacity of working with hundreds of variables to solve the VRP in a couple of minutes with a good quality compared with the best-known solutions (BKS) [2]. This means in small scenarios, this is a viable solution. However, as the e-market is expanding worldwide the requirement will be that of solving thousands of variables with good quality and in a reasonable time. The increment in variables number has a tendency to increase exponentially the time required to solve these problems on classical computers. This is an aspect that quantum computers can overcome, encoding large problems in a reasonable time and solving them much faster with quantum algorithms such as the quantum approximate optimization algorithm (QAOA).

We foresee a future where quantum computers will reduce largely the costs associated with storage and transportation at a big scale thanks to the combination of forecasting backorders and route optimization.

## Proof of Concept

To show the applicability of our concept, we have used real data of some products in the [dataset backorder](https://github.com/akhiilkasare/Back-Order-Prediction-iNeuron). The dataset contains 23 variables including:

|     |     |     | | | |
| --- | --- | --- | - | - | - |
| **sku** Product ID | **national_inv** Current inventory level for the part |**lead_time** Transit time for product | **in_transit_qty** Amount of product in transit from source|**forecast_3_month** Forecast sales for the next 3 months | **forecast_6_month** Forecast sales for the next 6 months|
|**forecast_9_month** Forecast sales for the next 9 months|**sales_1_month** Sales quantity for the prior 1 month time period|**sales_3_month** Sales quantity for the prior 3 month time period |**sales_6_month** Sales quantity for the prior 6 month time period |**sales_9_month** Sales quantity for the prior 9 month time period |**min_bank** Minimum recommend amount to stock 
|**potential_issue** Source issue for part identified |**pieces_past_due** Parts overdue from source |**perf_6_month_avg** Source performance for prior 6 month period |**perf_12_month_avg** Source performance for prior 12 month period |**local_bo_qty** Amount of stock orders overdue|**deck_risk** Part risk flag
|**oe_constraint** Part risk flag | **ppap_risk** Part risk flag | **stop_auto_buy** Part risk flag | **rev_stop** Part risk flag| **went_on_backorder** – Product actually went on backorder. <font color='green'>This is the target value</font>.


The dataset contains 1687861 products of some companies with a portion of 11293 that went backorder. We take 1000 cases equally distributed between <font color='green'>True</font> and <font color='red'>False</font> backorders from the dataset to do the training of our QNN. Once, we can predict if a product is backorder we select a small dataset of backorder products to make an optimization using the VRP with QAOA and VQE to select the optimal route. The comparison of the VRP results are contrasted with those of [docplex](https://pypi.org/project/docplex/) a classical optimizer. 

# Outline

1. Backorder prediction using a QNN


2. Vehicle routing problem solution. 


3. Conclusion and Future Work


4. References



- [Forecasting product demand](https://www.forbes.com/sites/amazonwebservices/2021/12/03/predicting-the-future-of-demand-how-amazon-is-reinventing-forecasting-with-machine-learning/?sh=7da9e49e1b6b) is our first step. We present a solution based on a quantum neural network (QNN) that combines classical machine learning layers and parametric quantum circuits to predict the behaviour of customers to allocate certain goods in strategic locations. Additionally, we present a solution for the delay of products. This will be connected directly with the solution 

- Solving the [Vehicle routing problem](https://en.wikipedia.org/wiki/Vehicle_routing_problem)(VRP) on real time. Important to work with thousands of variables in real time, current methods maximum couple hundreds

# 2. Vehicle routing problem Solution

Once we have the solution of the backorders predictions we select some of them to do an optimization of the best route giving a set of *K* trucks. The cost function associated to this problem is given by:


<img src="./Images/CostFunc.png" width=1000>


# 3. Conclusion and Future Work

We want to improve the restriction in our model time of importing and exporting, tendency of sell it in differente zones of the country.


# 4. References

[1] Psaraftis, H.N. (1988). Vehicle Routing: Methods and Studies. 16: 223–248.

[2] Tan, S., & Yeh, W. (2021). applied sciences The Vehicle Routing Problem : State-of-the-Art Classification and Review. Applied Sciences, 11.
