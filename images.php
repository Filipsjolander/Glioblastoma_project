<?php
	/*This is a 
	multi-line comment*/

	//This is a single-line comment

	#This is also a singe-line comment

	$user = 'root';
	$password = 'dagsforlunch222';
	$db = 'image_information_manager';
	$host = 'ip-172-31-6-241';
	$port = 3306;
	$socket = '/var/run/mysqld/mysqld.sock';

	$link = mysqli_init();
	$success = mysqli_real_connect(
  	 $link,
   	$host,
   	$user,
   	$password,
   	$db,
   	$port,
   	$socket
	);


	$sql = "SELECT * FROM `pictures_skincancers_project2`";
	$result = mysqli_query($link, $sql);

	
	for ($set = array (); $row = $result->fetch_assoc(); $set[] = $row);

	$number_of_pictures = count($set);
	
	$a = rand(1, ($number_of_pictures));
	$b = rand(1, ($number_of_pictures));
	$c = rand(1, ($number_of_pictures));
	if($b == $a){
		$b = (($b+1)%$number_of_pictures)+1;
	}
	while ($c == $a | $c == $b){
		$c = (($c+1)%$number_of_pictures)+1;
	}
	$link_a = $set[($a-1)]["link"];
	$link_b = $set[($b-1)]["link"];
	$link_c = $set[($c-1)]["link"];
?>

<!DOCTYPE html>
<html>
<head>
	<title> Image comparison. </title>
	<link rel="stylesheet" type="text/css" href="css2/stylesheet.css">
</head>
<body>
	<div id=page-form2>

		<section class="header">
  			<div class="title-form">
  				<h1>
					Welcome to the Nelander lab
				</h1>
				<h3>
					World leaders in cancer research.
				</h3>
  			</div>

  			<div class="logos">
  				<img src="images/SLL_logo-1024x331.png" 
  				style= '
  				margin-right: 10px;
				margin-top: 55px;
				margin-left: 100px; 
				width: 120px'
				alt="SciLifeLab logo">

  				<img src="images/uppsala.jpg" width="160" alt="Uppsala university logo">
  				

  				
  			</div>
		</section>

		<div id=text-form>
			<?php
				if(!empty($_POST["radio"])){
					echo "<div id=message>";
					$str = $_POST["radio"];
					$variable = explode(" ",$str);

					$A = $variable[0];
					$B = $variable[1];
					$C = $variable[2];
		
					#echo "The most unlike picture out of $A, $B and $C was $C."; 

					$sql = "INSERT INTO `data_points_project2` (`id`, `similar1`, `similar2`, `dissimilar`) VALUES (NULL, '$A', '$B', '$C')";

					#echo $sql;

					if (mysqli_query($link, $sql)) {
				 		echo " New record created successfully.";
					} 
					else {
				 		echo "Error: " . $sql . "<br>" . mysqli_error($link);
					}
					echo "</div>";
				}					
			?>
			<p>
				You are comparing the pictures
				<?php

					echo " A, B, and C.<br>";
				?>
			</p>

			<p>
				Which picture is the most unlike the others?
			</p>

			<div style="margin: 20px;">
				<div style="float: left; width: 286px;">
					A: 
					<?php 
					#echo $link_a;
					echo "<br>";
					echo '<img src="Project2_fyra_typer/'.$link_a.'" width="280px" alt="Uppsala university logo">';
					echo "<form action='' method='post'>
    						<button name='radio' value='$c $b $a'>A</button>
						</form>";
					?>

				</div>

				<div style="float: left; width: 286px;">
					B: 
					<?php 
					#echo $link_b;
					echo "<br>";
					echo '<img src="Project2_fyra_typer/'.$link_b.'" width="280px" alt="Uppsala university logo">';
					echo "<form action='' method='post'>
    						<button name='radio' value='$c $a $b'>B</button>
						</form>";
					?>



				</div>
				<div style="float: left; width: 286px;">
					C: 
					<?php 
					#echo $link_c;
					echo "<br>";
					echo '<img src="Project2_fyra_typer/'.$link_c.'" width="280px" alt="Uppsala university logo">';
					echo "<form action='' method='post'>
    						<button name='radio' value='$a $b $c'>C</button>
						</form>";
					?>

				</div>
				 <br style="clear: left;" />
			</div>

		</div>
		<br>
		<a href="index.html">Back to home page.</a>
	</div>

</body>
</html>








