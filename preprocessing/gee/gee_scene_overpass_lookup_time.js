/**
 * Pull scene-center acquisition times for all scenes in the lookup table.
 * 
 * Approach: filter each Landsat collection by system:index to avoid
 * the "parameter id must be a constant" error from ee.Image(dynamic_id).
 */

// ==============================
// CONFIG
// ==============================
// LOOKUP_ASSET points to a private Earth Engine asset. Replace it with your own
// uploaded FeatureCollection (same schema: gee_id, sat_id, mission) before running.
// To build that table from preprocessing/join_aquamatch_sitesr.py's (Stage 1) output,
// see data/README.md -- gee_id is not a native siteSR column and has to be
// reconstructed from sat_id and mission.
var LOOKUP_ASSET = 'projects/ee-my-atuobimsc24/assets/scene_lookup';
var EXPORT_DESC = 'scene_center_times';
var EXPORT_FOLDER = 'EarthEngine';

// ==============================
// LOAD LOOKUP TABLE
// ==============================
var lookup = ee.FeatureCollection(LOOKUP_ASSET);
print('Total scenes in lookup:', lookup.size());

// ==============================
// HELPER: extract scene index from gee_id
// e.g. "LANDSAT/LC08/C02/T1_L2/LC08_003048_20130519" -> "LC08_003048_20130519"
// ==============================
function getSceneIndex(gee_id) {
  return ee.String(gee_id).split('/').get(-1);
}

// ==============================
// PROCESS EACH MISSION
// ==============================
var missions = {
  'LT04': ee.ImageCollection('LANDSAT/LT04/C02/T1_L2'),
  'LT05': ee.ImageCollection('LANDSAT/LT05/C02/T1_L2'),
  'LE07': ee.ImageCollection('LANDSAT/LE07/C02/T1_L2'),
  'LC08': ee.ImageCollection('LANDSAT/LC08/C02/T1_L2'),
  'LC09': ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'),
};

var allResults = ee.FeatureCollection([]);

var missionKeys = ['LT04', 'LT05', 'LE07', 'LC08', 'LC09'];

missionKeys.forEach(function(missionKey) {
  var collection = missions[missionKey];
  
  // Get lookup rows for this mission
  var missionLookup = lookup.filter(ee.Filter.eq('mission', missionKey));
  var count = missionLookup.size();
  
  // Extract scene indices for filtering
  var sceneIndices = missionLookup.aggregate_array('gee_id').map(function(id) {
    return getSceneIndex(id);
  });
  
  // Filter the Landsat collection to only our scenes
  var filtered = collection.filter(ee.Filter.inList('system:index', sceneIndices));
  
  // Extract time properties from each matched scene
  var withProps = filtered.map(function(image) {
    var sysIndex = image.get('system:index');
    var timeStart = image.get('system:time_start');
    var acquisitionDT = ee.Date(timeStart).format('YYYY-MM-dd HH:mm:ss', 'UTC');
    var sceneCenterTime = image.get('SCENE_CENTER_TIME');
    var dateAcquired = image.get('DATE_ACQUIRED');
    
    return ee.Feature(null, {
      'system_index': sysIndex,
      'mission': missionKey,
      'acquisition_datetime_utc': acquisitionDT,
      'system_time_start_ms': timeStart,
      'scene_center_time': sceneCenterTime,
      'date_acquired': dateAcquired
    });
  });
  
  allResults = allResults.merge(withProps);
  
  print(missionKey + ' lookup rows:', count);
  print(missionKey + ' matched scenes:', filtered.size());
});

// ==============================
// JOIN BACK TO LOOKUP TABLE TO GET sat_id
// ==============================
// The lookup table has gee_id, the results have system_index
// We need to add system_index to lookup for the join
var lookupWithIndex = lookup.map(function(feature) {
  var gee_id = ee.String(feature.get('gee_id'));
  var sysIndex = getSceneIndex(gee_id);
  return feature.set('system_index', sysIndex);
});

// Join on system_index
var joinFilter = ee.Filter.equals({
  leftField: 'system_index',
  rightField: 'system_index'
});

var join = ee.Join.inner('lookup', 'scene');
var joined = join.apply(lookupWithIndex, allResults, joinFilter);

// Flatten the joined features
var final = joined.map(function(feature) {
  var lookupF = ee.Feature(feature.get('lookup'));
  var sceneF = ee.Feature(feature.get('scene'));
  return ee.Feature(null, {
    'sat_id': lookupF.get('sat_id'),
    'gee_id': lookupF.get('gee_id'),
    'mission': sceneF.get('mission'),
    'acquisition_datetime_utc': sceneF.get('acquisition_datetime_utc'),
    'system_time_start_ms': sceneF.get('system_time_start_ms'),
    'scene_center_time': sceneF.get('scene_center_time'),
    'date_acquired': sceneF.get('date_acquired')
  });
});

// ==============================
// CHECK
// ==============================
print('Final matched count:', final.size());
print('Preview:', final.limit(5));

// ==============================
// EXPORT
// ==============================
Export.table.toDrive({
  collection: final,
  description: EXPORT_DESC,
  folder: EXPORT_FOLDER,
  fileFormat: 'CSV',
  selectors: [
    'sat_id', 'gee_id', 'mission',
    'acquisition_datetime_utc', 'system_time_start_ms',
    'scene_center_time', 'date_acquired'
  ]
});
