
import numpy as np

def calcPlanetPosition(local_latitude, local_longitude, year, month, day, time, planet):

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

	E = M + e*(180/np.pi) * np.sin(M) * ( 1.0 + e * np.cos(M) )

	xv = np.cos(E) - e
	yv = np.sqrt(1.0 - e*e) * np.sin(E)

	v = np.arctan2( yv, xv )
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


	E = M + e * np.sin(M) * ( 1.0 + e * np.cos(M) )

	if(e > 0.055):
		E1 = 0.0
		E0 = E
		j = 0
		while E1 - E0 > 0.001 or E0 - E1 > 0.001:
			if(j != 0):
				E0 = E1
			E1 = E0 - ( E0 - e*(180/np.pi) * np.sin(E0) - M ) / ( 1 - e * np.cos(E0) )
			j += 1
			if j >= 99999:
				print("NOOOOOOOO TOO MANY ITERATIONS USE OTHER FORMULA SAD")
				return
		E = E1

	xv = a * ( np.cos(E) - e )
	yv = a * ( np.sqrt(1.0 - e*e) * np.sin(E) )

	v = np.arctan2( yv, xv )
	r = np.sqrt( xv*xv + yv*yv )

	xh = r * (np.cos(N) * np.cos(v+w) - np.sin(N) * np.sin(v+w) * np.cos(i))
	yh = r * (np.sin(N) * np.cos(v+w) + np.cos(N) * np.sin(v+w) * np.cos(i))
	zh = r * (np.sin(v+w) * np.sin(i) )

	lonecl = np.arctan2( yh, xh )
	latecl = np.arctan2( zh, np.sqrt(xh*xh+yh*yh) )

	if planet == "Jupiter":
		Ms = 316.9670 + 0.0334442282 * d
		Mj = M
		lonecl -= 0.332 * np.sin(2*Mj - 5*Ms - 67.6)
		lonecl -= 0.056 * np.sin(2*Mj - 2*Ms + 21)
		lonecl += 0.042 * np.sin(3*Mj - 5*Ms + 21)
		lonecl -= 0.036 * np.sin(Mj - 2*Ms)
		lonecl += 0.022 * np.cos(Mj - Ms)
		lonecl += 0.023 * np.sin(2*Mj - 3*Ms + 52)
		lonecl -= 0.016 * np.sin(Mj - 5*Ms - 69)

	if planet == "Saturn":
		Mj = 19.8950 + 0.0830853001 * d
		Ms = M
		lonecl += 0.812 * np.sin(2*Mj - 5*Ms - 67.6)
		lonecl -= 0.229 * np.cos(2*Mj - 4*Ms - 2)
		lonecl += 0.119 * np.sin(Mj - 2*Ms - 3)
		lonecl += 0.046 * np.sin(2*Mj - 6*Ms - 69)
		lonecl += 0.014 * np.sin(Mj - 3*Ms + 32)

		latecl -= 0.020 * np.cos(2*Mj - 4*Ms - 2)
		latecl += 0.018 * np.sin(2*Mj - 6*Ms - 49)

	xh = r * np.cos(lonecl) * np.cos(latecl)
	yh = r * np.sin(lonecl) * np.cos(latecl)
	zh = r * np.sin(latecl)

	xs = rs * np.cos(lonsun)
	ys = rs * np.sin(lonsun)

	xg = xh + xs
	yg = yh + ys
	zg = zh

	xe = xg
	ye = yg * np.cos(ecl) - zg * np.sin(ecl)
	ze = yg * np.sin(ecl) + zg * np.cos(ecl)

	RA  = np.arctan2( ye, xe )
	Decl = np.arctan2( ze, np.sqrt(xe*xe+ye*ye) )

	HA = sun_LST - RA

	x = np.cos(HA) * np.cos(Decl)
	y = np.sin(HA) * np.cos(Decl)
	z = np.sin(Decl)

	xhor = x * np.sin(local_latitude) - z * np.cos(local_latitude)
	yhor = y
	zhor = x * np.cos(local_latitude) + z * np.sin(local_latitude)

	az  = np.arctan2( yhor, xhor ) + 180
	alt = np.arctan2( zhor, np.sqrt(xhor*xhor+yhor*yhor) )

	print(az)
	print(alt)
	return az, alt


if __name__=="__main__": 
    calcPlanetPosition(22, 77, 1900, 3, 15, 8, "Mercury")







	





