
import numpy as np

def calcPlanetPosition(local_latitude, local_longitude, year, month, day, time, planet):

	print(planet)

	d = 367*year - 7 * (year + (month+9)/12 ) / 4 - 3 * ((year + (month-9)/7 ) / 100 + 1 ) / 4 + 275*month/9 + day - 730515
	d = d + time / 24.0

	N = 0.0
	i = 0.0
	w = 0.0
	a = 0.0
	e = 0.0
	M = 0.0
	ecl = 23.4393 - 3.563E-7 * d

	sun_N = 0.0
	sun_i = 0.0
	sun_w = 282.9404 + 4.70935E-5 * d
	sun_a = 1.000000
	sun_e = 0.016709 - 1.151E-9 * d
	sun_M = 356.0470 + 0.9856002585 * d

	E = M + e*(180/np.pi) * degreeSin(M) * ( 1.0 + e * degreeCos(M) )

	xv = degreeCos(E) - e
	yv = np.sqrt(1.0 - e*e) * degreeSin(E)

	v = degreeArctan2( yv, xv )
	rs = np.sqrt( xv*xv + yv*yv )

	lonsun = v + sun_w



	sun_Ls = sun_M +sun_w
	sun_GMST0 = sun_Ls/15 + 12
	sun_GMST = sun_GMST0 + time
	sun_LST  = sun_GMST + local_longitude/15

	if planet == "Mercury":
	 	N =  48.3313 + 3.24587E-5 * d
	 	i = 7.0047 + 5.00E-8 * d
	 	w =  29.1241 + 1.01444E-5 * d
	 	a = 0.387098
	 	e = 0.205635 + 5.59E-10 * d
	 	M = 168.6562 + 4.0923344368 * d
	elif planet == "Venus":
		N =  76.6799 + 2.46590E-5 * d
		i = 3.3946 + 2.75E-8 * d
		w =  54.8910 + 1.38374E-5 * d
		a = 0.723330
		e = 0.006773 - 1.302E-9 * d
		M =  48.0052 + 1.6021302244 * d
	elif planet == "Mars":
		N =  49.5574 + 2.11081E-5 * d
		i = 1.8497 - 1.78E-8 * d
		w = 286.5016 + 2.92961E-5 * d
		a = 1.523688
		e = 0.093405 + 2.516E-9 * d
		M =  18.6021 + 0.5240207766 * d
	elif planet == "Jupiter":
		N = 100.4542 + 2.76854E-5 * d
		i = 1.3030 - 1.557E-7 * d
		w = 273.8777 + 1.64505E-5 * d
		a = 5.20256
		e = 0.048498 + 4.469E-9 * d
		M =  19.8950 + 0.0830853001 * d
	elif planet == "Saturn":
		N = 113.6634 + 2.38980E-5 * d
		i = 2.4886 - 1.081E-7 * d
		w = 339.3939 + 2.97661E-5 * d
		a = 9.55475
		e = 0.055546 - 9.499E-9 * d
		M = 316.9670 + 0.0334442282 * d


	E = M + e * degreeSin(M) * ( 1.0 + e * degreeCos(M) )

	if(e > 0.055):
		E1 = 0.0
		E0 = E
		j = 0
		while E1 - E0 > 0.001 or E0 - E1 > 0.001:
			if(j != 0):
				E0 = E1
			E1 = E0 - ( E0 - e*(180/np.pi) * degreeSin(E0) - M ) / ( 1 - e * degreeCos(E0) )
			j += 1
			if j >= 99999:
				print("NOOOOOOOO TOO MANY ITERATIONS USE OTHER FORMULA SAD")
				return
		E = E1

	xv = a * ( degreeCos(E) - e )
	yv = a * ( np.sqrt(1.0 - e*e) * degreeSin(E) )

	v = degreeArctan2( yv, xv )
	r = np.sqrt( xv*xv + yv*yv )

	xh = r * (degreeCos(N) * degreeCos(v+w) - degreeSin(N) * degreeSin(v+w) * degreeCos(i))
	yh = r * (degreeSin(N) * degreeCos(v+w) + degreeCos(N) * degreeSin(v+w) * degreeCos(i))
	zh = r * (degreeSin(v+w) * degreeSin(i) )

	lonecl = degreeArctan2( yh, xh )
	latecl = degreeArctan2( zh, np.sqrt(xh*xh+yh*yh) )

	if planet == "Jupiter":
		Ms = 316.9670 + 0.0334442282 * d
		Mj = M
		lonecl -= 0.332 * degreeSin(2*Mj - 5*Ms - 67.6)
		lonecl -= 0.056 * degreeSin(2*Mj - 2*Ms + 21)
		lonecl += 0.042 * degreeSin(3*Mj - 5*Ms + 21)
		lonecl -= 0.036 * degreeSin(Mj - 2*Ms)
		lonecl += 0.022 * degreeCos(Mj - Ms)
		lonecl += 0.023 * degreeSin(2*Mj - 3*Ms + 52)
		lonecl -= 0.016 * degreeSin(Mj - 5*Ms - 69)

	if planet == "Saturn":
		Mj = 19.8950 + 0.0830853001 * d
		Ms = M
		lonecl += 0.812 * degreeSin(2*Mj - 5*Ms - 67.6)
		lonecl -= 0.229 * degreeCos(2*Mj - 4*Ms - 2)
		lonecl += 0.119 * degreeSin(Mj - 2*Ms - 3)
		lonecl += 0.046 * degreeSin(2*Mj - 6*Ms - 69)
		lonecl += 0.014 * degreeSin(Mj - 3*Ms + 32)

		latecl -= 0.020 * degreeCos(2*Mj - 4*Ms - 2)
		latecl += 0.018 * degreeSin(2*Mj - 6*Ms - 49)

	xh = r * degreeCos(lonecl) * degreeCos(latecl)
	yh = r * degreeSin(lonecl) * degreeCos(latecl)
	zh = r * degreeSin(latecl)

	xs = rs * degreeCos(lonsun)
	ys = rs * degreeSin(lonsun)

	xg = xh + xs
	yg = yh + ys
	zg = zh

	xe = xg
	ye = yg * degreeCos(ecl) - zg * degreeSin(ecl)
	ze = yg * degreeSin(ecl) + zg * degreeCos(ecl)

	RA  = degreeArctan2( ye, xe )
	Decl = degreeArctan2( ze, np.sqrt(xe*xe+ye*ye) )

	HA = sun_LST - RA

	x = degreeCos(HA) * degreeCos(Decl)
	y = degreeSin(HA) * degreeCos(Decl)
	z = degreeSin(Decl)

	xhor = x * degreeSin(local_latitude) - z * degreeCos(local_latitude)
	yhor = y
	zhor = x * degreeCos(local_latitude) + z * degreeSin(local_latitude)

	az  = degreeArctan2( yhor, xhor ) + 180
	alt = degreeArctan2( zhor, np.sqrt(xhor*xhor+yhor*yhor) )

	print(az)
	print(alt)
	return az, alt


def degreeCos(deg):
	return np.cos(np.deg2rad(deg))

def degreeSin(deg):
	return np.sin(np.deg2rad(deg))

def degreeArctan2(val1, val2):
	return np.rad2deg(np.arctan2(val1, val2))



if __name__=="__main__": 
    calcPlanetPosition(20.5937, 78.9629, 496, 1, 1, 0, "Jupiter")










	





