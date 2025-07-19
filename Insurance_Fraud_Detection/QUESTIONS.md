# Written Questions
## Question 1: LLM Integration Strategy
Design how you would integrate LLM into this fraud detection system: 

__How would LLMs analyze claim descriptions for fraud indicators?__  
_I would check the grammar, typos, style. Mistakes and typos can indicate that it is a fraud. Style that is consistend with datasets of fraudulent activities indicate the same. Would also check if the text description matches with the numeric fields in the CSV._

__What prompt engineering approach would you use for insurance claims?__  
_Will define the LLM's "role" as an insurance analyst. Provide "context": that it is working on an insurance fraud identification task. Maybe provide a couple examples ("2-shot approach"). Will ask it to reason "Chain-Of-Thought" so that it checks itself as it goes. Ask it to provide a structured output (and use Pydantic library) with a Fraud/NoFraud binary True/False output, and maybe it's reasoning as a string in another output field (for further manual review)._

__Which LLM would you choose (GPT-4, Claude, etc.) and why?__
_I would choose GPT-4 for the consistency in its performance._

## Question 2: Production Deployment Architecture (7 minutes)
Describe your approach to deploying this model in production on any cloud provider (e.g., AWS, Azure, GCP):


__What services or components would you use to deploy and scale the model, and why?__
_I'd use AWS SageMaker for the deployment - scalable, easier infrastructure management and relatively cheaper than self-managing EC2. If LLM is added later (as discussed in the previous question) - will lok into using AWS Bedrock for deployment._

__How would you handle real-time scoring for 100K+ daily insurance claims?__  
_AWS SageMaker is highly scalable, should handle the inference well. regarding the claims ingestion will look into using Apache kafka._

__What strategies would you use to monitor the system and detect model drift over time?__  
_Will monitor the number of fraud/nofraud classified examples per day and as a moving average to see if there is a significant shift. If the system is deployed and used in production, then probably the claims are manually reviewed and audited as well. Will keep comparing the metrics Accuracy, Recall over time. Will also look at the feature distributions over time (like the claim sizes, etc.)_
