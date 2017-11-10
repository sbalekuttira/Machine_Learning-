Requirements:
1)datatype.py
2)channel.py
3)datatype dataset (12 months)
4)channel dataset (12 months)
5)cost.txt (cost data)



->channel.py

What the code does?
1)It reads all the 12 months of channel data
2)It plots a graph with points being number of violations of each type of each month 
3)I have re written Linear Regression Best Fit line code by myself. So it is self explantory . It draws the regression line for the graph
4)In the predict code you can predict any months number of violation 
5)last year january stands for 1 and next year january stands for 13 , so if i want to predict next three months put x=13 , x=14 , x=15
6) it will calucalte the Y (number of predicted violations) and it will plot it on the graph with a differnt color
7) by default right now it predicts one month . Next year january for all types
8) you can see that point as a differnt color in each graph 
9)it uses cost.txt to find the cost for every violation in january and puts into the summary file 


Note: Code explanations are put in comments in the code itself



->datatype.py

What the code does?
1)It reads all the 12 months of datatype data
2)It plots a graph with points being number of violations of each type of each month 
3)I have re written Linear Regression Best Fit line code by myself. So it is self explantory . It draws the regression line for the graph
4)In the predict code you can predict any months number of violation 
5)last year january stands for 1 and next year january stands for 13 , so if i want to predict next three months put x=13 , x=14 , x=15
6) it will calucalte the Y (number of predicted violations) and it will plot it on the graph with a differnt color
7) by default right now it predicts one month . Next year january for all types
8) you can see that point as a differnt color in each graph 
9)it uses cost.txt to find the cost for every violation in january and puts into the summary file 


Note: Code explanations are put in comments in the code itself
      variable names are expanded in the code comments


->estimated_cost_january.txt

what does it have?
1) has all the cost 
2)self explantory 
3)has the total cost as well


->cost.csv
1)it has all the costs for each type of violation 
the order is [email,web,usb,print,PersonalIdentity info,Personal Credit info,Internal Docs,Design Docs,Source Code,Inventory Disclosure,Program restricted Data]

you can change the costs if required. But maintain the order.

->cost_inter.txt

why required?
1)just an intermediate file to send cost from datatype.py to channel.py to find total cost for january 


->sequence to execute
1)make sure there is no file of the name estimated_cost_january.txt (if its there from previous execution , make sure to delete)
2)make sure there is no file of the name cost_inter.txt (if its there from previous execution , make sure to delete)
3)run datatype.py
4)explain the graphs
5)show the text file estimated_cost_january.txt
6)run channel.py
7)explain the graphs
8)show the text file estimated_cost_january.txt
9)THE END


