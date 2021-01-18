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

-- 6. Display countries that have the world 'United' in their name
SELECT name
FROM world
WHERE name LIKE '%united';