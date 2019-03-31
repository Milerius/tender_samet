![first_page](assets/face_page.png)

# 1. SYSTEM PRESENTATION

## 1.1 CONTEXT

This document presents the specification for the scheduling module that transforms prediction output from lung radio images, into an appointment with a lung specialist.

After disease detection and validation by a radiologist, a patient is given a priority score depending on the severity of his disease, as well as a recommended date range for a follow-up appointment. Here, the date range and priority score are parameters provided by medical professionals and are not deduced from machine learning. The process of assigning priority and date can eventually be partially automated as we will see in section 3, but a human in the loop is still preferred for this step.

After receiving his or her score, the result is sent to the Customer Relationship Management (**CRM**) of the patient’s hospital. Taking into account the necessary date range and priority of treatment he/she gets an appointment made with a doctor.

The advantage of this approach is that it prioritizes the most urgent cases and helps doctors manage their schedule efficiently.

## 1.2 SCHEDULING MODULE

![schedule_module](assets/schedule_module.png)

The scheduling module acts as an interface between the hospital **CRM** and various instances (an instance per doctor) of the **AI** detections systems

The **AI** instances send their detection results to the scheduler in **JSON** format. Each **JSON** contains the following fields:

| **FIELD**                  	| **TYPE**         	| **DESCRIPTION**                                                                  	|
|------------------------	|--------------	|------------------------------------------------------------------------------	|
| **NAME**                   	| String       	| Name of the patient                                                          	|
| **PATIENT_ID**             	| Number       	| Unique ID of patient                                                         	|
| **PRIORITY_SCORE**         	| Number       	| How urgent it is to prioritize the patient in the queue                      	|
| **RECOMMENDED_DATE_RANGE** 	| [DATE, DATE] 	| The date range when it is recommended for the patient to get next,treatment. 	|

The **CRM** also sends information to the module with patients who want to cancel their appointments because the date and time are not convenient for them. It is also in **JSON** format:

| **FIELD**                  	| **TYPE**        	| **DESCRIPTION**                                                                      	|
|------------------------	|-------------	|----------------------------------------------------------------------------------	|
| **NAME**                   	| String      	| Name of the patient                                                              	|
| **PATIENT_ID**             	| Number      	| Unique ID of patient                                                             	|
| **CURRENT_DATE**           	| DATE        	| The date of the appointment that the patient wants to cancel                     	|
| **NUMBER_CANCELLATIONS**   	| Number      	| The number of times the patient has already cancelled his/her,appointment before 	|
| **RECOMMENDED_DATE_RANGE** 	| [Date,Date] 	| The date range when it is recommended for the patient to get treatment.          	|
| **PRIORITY_SCORE**         	| Number      	| How urgent it is to prioritize the patient in the queue                          	|

The scheduler creates a **JSON** file where it stores all the `JSON objects` that receives. The format is the same as the one immediately above.

The scheduling systems works in daily batch mode. After 9pm the current json file archive is disconnected from the data flux and replaced with another one. The scheduler then enters scheduling mode, where it receives the list of doctor schedules from the **CRM**, runs a prioritization algorithm and returns a new set of schedules to the **CRM**.

## 1.3 THE CRM

For the system to work. The **CRM** must provide an external API that enables the module to get object-relational mapping (ORM) representations of the data model used by the **CRM**. It should be able to provide features like get and post in a commonly used programming language like python or Java. This would allow for encapsulation and makes sure that the scheduling module remains independent of the **CRM** implementation details.

One CRM that we found that provided such a capability was the **Odoo CRM**: https://www.odoo.com

It is not only open source and extendable, but it also provides an API on both Python and Java: https://www.odoo.com/documentation/12.0/webservices/odoo.html

The **CRM** must also be able to be extendable by modules that allow it to send special purpose files containing information about patients who wish to cancel their appointments.

Finally, it must send a notification to the patients via their email and/or phone number and allow for a smooth user experience from the patient side in terms of appointment cancellation requests. 

## 1.4 THE AI MODULE

The AI module is composed of both the ML prediction engine and, in the first version, a human in the loop who assigns to each diagnostic a priority score and a recommended date range for treatment.

In order to mitigate the possibility of false positives, a radiologist is asked to validate the model prediction at the end. 

In order to mitigate the risk of false negatives (patients with an illness flagged as healthy) we still apply a follow-up strategy to reduce the risk of untreated diseases becoming a severe health issue. 

These mitigation strategies incur a cost in terms of appointment scheduling. A true negative means that a follow up interview is useless, same thing for a false positive.

A False negative however if not detected and given a follow up appointment might cost many sessions in case of complications, and if urgent hospitalization is needed, then cost can be high.
The cost-benefit equation can be written as:

`Cost = (FP + TN) * cost_of_mitigation – FN * cost_of_untreated_illness`

With:

* FP: False positive rate
* TN: True negative rate
* FN: False negative rate

Given how these rates are influenced by the detection threshold of the classification model. One can use the ROC curve to determine what threshold to use to so that the cost does not go to high.

