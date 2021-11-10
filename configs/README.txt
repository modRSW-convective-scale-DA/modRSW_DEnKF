
###################### Q MATRIX COMPUTATION ##############################

### CONFIG NUMBER 1-8,13-15:	 		Q is diagonal with r variance non zero (based on 1-hour errors)
### CONFIG NUMBER 9-12:  			Q is diagonal with r vairance zero (based on 1-hour errors)
### CONFIG NUMBER 16-17,20,23-24:       	Q is diagonal with r variance non zero (based on 6-hours error)
### CONFIG NUMBER 18-19,21-22,25-35,38-39:	Q is predetermined via fixed model_noise from config file
### CONFIG NUMBER 36-37:			Q is diagonal with r variance non zero and u variance zero (based on 6-hours error)
### CONFIG NUMBER 57-60:			Q is diagonal with r variance non zero (based on 3-hours error)

FROM CONFIG 64:

Q is diagonal with r variance non zero (based on 1-hour error)
