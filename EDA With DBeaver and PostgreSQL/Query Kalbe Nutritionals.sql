---Query 1 "rata-rata umur customer bedasarkan marital status"
select "Marital Status", avg(age) as "Rata-rata Usia" 
from customer 
group by "Marital Status";

---Query 2 "rata-rata umur customer berdasarkan gender"
select gender , avg(age) as "Rata-rata Usia" 
from customer 
group by gender;

---Query 3 "nama store dengan total quantity terbanyak"
select store.storename , sum("Transaction".qty) as "Total Quantity"
from store inner join "Transaction" 
on store.storeid = "Transaction".storeid 
group by store.storename 
order by "Total Quantity" desc 
limit 1;

---Query 4 "nama produk terlaris dengan total amount terbanyak"
select product."Product Name" , sum("Transaction".totalamount) as "Total Amount"
from product inner join "Transaction" 
on product.productid = "Transaction".productid 
group by product."Product Name" 
order by "Total Amount" desc 
limit 1;