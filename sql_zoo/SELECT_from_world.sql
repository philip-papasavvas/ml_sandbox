-- Answers to exercises from https://sqlzoo.net/wiki/SELECT_from_WORLD_Tutorial

-- 1. Select the name, continent, and population of all countries
SELECT name, continent, population
FROM world;

-- 2. Select countries with more than 200 million population
SELECT name, gdp/population
FROM  world
WHERE population > 200000000; -- quite messy but this is 200 with 6 zeros after

-- 3. Display GDP per capita for countries with more than 200 million population
SELECT name, gdp/population
FROM world
WHERE population > 200000000;

-- 4. Display name, population (in millions) for countries in the continent 'South America'
SELECT name, population/1000000
FROM world
WHERE continent = 'South America';

-- 5. Display the name, population for France, Germany, Italy
SELECT name, population
FROM world
WHERE name IN ('France', 'Germany', 'Italy');

-- 6. Display countries that have the word 'United' in their name
SELECT name
FROM world
WHERE name LIKE '%united';

-- 7. Show the countries (name, population, area) that are big by area (>3 million sq km)
--      or big by population (> 250 million).
SELECT name, population, area
FROM world
WHERE (area > 3000000)
OR (population > 250000000);

-- 8. Exclusive OR (XOR). Show the countries (name, population, area) that are big by area
-- (>3 million sq km) or big by population (> 250 million), but not both.
SELECT name, population, area
FROM world
WHERE (area > 3000000 AND population < 250000000)
OR (population > 250000000 AND area < 3000000);

-- 9 . Show the name and population (in millions) and GDP (in billions) for countries from the
-- continent 'South America'
SELECT name
 ,ROUND(population/1000000, 2)
 ,ROUND(GDP/1000000000, 2)
FROM world
WHERE continent = 'South America';

/* 10. Show the name and per-capita GDP for countries with GDP of more than 1 trillion (12 zeros)
    round the value to the nearest 1000.
     # note the ROUND(f,p) returns f rounded to p decimal places. these can be negative, round
     to nearest 10 when p is -1, nearest 100 for p = -2, etc */
SELECT name, ROUND(gdp/population, -3)
FROM world
where gdp > 1000000000000;

/* 11. Show the name and capital where the name and capital both have the same number of characters */
SELECT name, capital
FROM world
WHERE length(name) = length(capital);
