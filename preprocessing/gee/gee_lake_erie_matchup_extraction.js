/**
 * Lake Erie external-validation Landsat matchup extraction.
 *
 * Field observations with valid local sampling dates and times are matched to
 * Landsat 5, 7, 8, and 9 Collection 2 Level-2 imagery. Candidate scenes are
 * searched within a +/-48 hour window of each field observation. The temporally
 * closest scene containing QA-valid pixels within a 200 m buffer is selected
 * first. Dynamic Surface Water Extent (DSWE) is then calculated on that same
 * scene, and the matchup is retained only when at least eight DSWE class-1
 * high-confidence open-water pixels occur within the buffer. Median surface
 * reflectance is calculated using only those water pixels.
 */


// sampleAsset points to a private Earth Engine asset. Replace it with your own uploaded
// FeatureCollection (same schema: the dateCol/timeCol/idCol/programCol/obsCol fields below)
// before running.
var CONFIG = {
  sampleAsset: 'projects/ee-my-atuobimsc24/assets/LakeErie_AllPrograms_Chl',

  dateCol: 'Date',
  timeCol: 'Time',
  idCol: 'ID#',
  programCol: 'Program',
  obsCol: 'Chlorophyll (µg/L)',

  timezone: 'America/New_York',
  datetimeFormat: 'M/d/yyyy H:mm',
  dateRegex: '^\\d{1,2}/\\d{1,2}/\\d{4}$',
  timeRegex: '^\\d{1,2}:\\d{2}$',

  timeWindowHours: 48,
  bufferRadiusMeters: 200,

  scale: 30,
  maxPixels: 1e9,
  maxSceneCloud: 80,
  minDswe1Pixels: 8,

  exportDescription:
    'LakeErie_AllPrograms_Landsat_Matchups_200m_DSWE1_ValidTime_FixedScene',

  exportFolder: 'EarthEngine'
};


function stringValue(feature, property) {
  return ee.String(
    ee.Algorithms.String(feature.get(property))
  ).trim();
}


var rawSamples = ee.FeatureCollection(CONFIG.sampleAsset);

var samples = rawSamples
  .filter(ee.Filter.notNull([
    CONFIG.dateCol,
    CONFIG.timeCol,
    CONFIG.idCol,
    CONFIG.obsCol
  ]))
  .map(function(feature) {
    var date = stringValue(feature, CONFIG.dateCol);
    var time = stringValue(feature, CONFIG.timeCol);

    var valid = date.match(CONFIG.dateRegex).length().gt(0)
      .and(time.match(CONFIG.timeRegex).length().gt(0));

    return feature.set('_valid_datetime', valid);
  })
  .filter(ee.Filter.eq('_valid_datetime', 1));


var studyArea = samples.geometry().bounds();


function sampleDateTime(feature) {
  var datetime = stringValue(feature, CONFIG.dateCol)
    .cat(' ')
    .cat(stringValue(feature, CONFIG.timeCol));

  return ee.Date.parse(
    CONFIG.datetimeFormat,
    datetime,
    CONFIG.timezone
  );
}


function maskAndScale(image) {
  var qa = image.select('QA_PIXEL');

  var qaMask = qa.bitwiseAnd(1 << 0).eq(0)
    .and(qa.bitwiseAnd(1 << 1).eq(0))
    .and(qa.bitwiseAnd(1 << 2).eq(0))
    .and(qa.bitwiseAnd(1 << 3).eq(0))
    .and(qa.bitwiseAnd(1 << 4).eq(0))
    .and(qa.bitwiseAnd(1 << 5).eq(0))
    .and(image.select('QA_RADSAT').eq(0));

  var optical = image.select('SR_B.*')
    .multiply(0.0000275)
    .add(-0.2);

  return image
    .addBands(optical, null, true)
    .updateMask(qaMask);
}


function dswe1Mask(image) {
  var blue = image.select('Blue');
  var green = image.select('Green');
  var red = image.select('Red');
  var nir = image.select('NIR');
  var swir1 = image.select('SWIR1');
  var swir2 = image.select('SWIR2');

  var mndwi = green.subtract(swir1)
    .divide(green.add(swir1));

  var mbsrv = green.add(red);
  var mbsrn = nir.add(swir1);

  var ndvi = nir.subtract(red)
    .divide(nir.add(red));

  var awesh = blue
    .add(green.multiply(2.5))
    .subtract(mbsrn.multiply(1.5))
    .subtract(swir2.multiply(0.25));

  var t1 = mndwi.gt(0.124);
  var t2 = mbsrv.gt(mbsrn);
  var t3 = awesh.gt(0);

  var t4 = mndwi.gt(-0.44)
    .and(swir1.lt(0.09))
    .and(nir.lt(0.15))
    .and(ndvi.lt(0.7));

  var t5 = mndwi.gt(-0.5)
    .and(blue.lt(0.1))
    .and(swir1.lt(0.3))
    .and(swir2.lt(0.1))
    .and(nir.lt(0.25));

  var code = t1
    .add(t2.multiply(10))
    .add(t3.multiply(100))
    .add(t4.multiply(1000))
    .add(t5.multiply(10000));

  var highConfidenceWater = code.eq(1111)
    .or(code.eq(10111))
    .or(code.eq(11011))
    .or(code.eq(11101))
    .or(code.eq(11110))
    .or(code.eq(11111));

  return highConfidenceWater
    .rename('DSWE1')
    .selfMask();
}


var commonBands = [
  'Blue',
  'Green',
  'Red',
  'NIR',
  'SWIR1',
  'SWIR2'
];


function buildCollection(collection, bands, sensor) {
  return ee.ImageCollection(collection)
    .filterBounds(studyArea)
    .filter(ee.Filter.lt('CLOUD_COVER', CONFIG.maxSceneCloud))
    .map(maskAndScale)
    .select(bands, commonBands)
    .map(function(image) {
      return image.set('sensor', sensor);
    });
}


var tmBands = [
  'SR_B1',
  'SR_B2',
  'SR_B3',
  'SR_B4',
  'SR_B5',
  'SR_B7'
];

var oliBands = [
  'SR_B2',
  'SR_B3',
  'SR_B4',
  'SR_B5',
  'SR_B6',
  'SR_B7'
];


var landsat = buildCollection(
  'LANDSAT/LT05/C02/T1_L2',
  tmBands,
  'Landsat_5_TM'
)
.merge(
  buildCollection(
    'LANDSAT/LE07/C02/T1_L2',
    tmBands,
    'Landsat_7_ETM+'
  )
)
.merge(
  buildCollection(
    'LANDSAT/LC08/C02/T1_L2',
    oliBands,
    'Landsat_8_OLI'
  )
)
.merge(
  buildCollection(
    'LANDSAT/LC09/C02/T1_L2',
    oliBands,
    'Landsat_9_OLI2'
  )
);


function qaPixelCount(image, region) {
  var result = image.select('Blue')
    .reduceRegion({
      reducer: ee.Reducer.count().unweighted(),
      geometry: region,
      scale: CONFIG.scale,
      maxPixels: CONFIG.maxPixels
    });

  return ee.Number(result.get('Blue', 0));
}


function waterPixelCount(mask, region) {
  var result = mask.reduceRegion({
    reducer: ee.Reducer.count().unweighted(),
    geometry: region,
    scale: CONFIG.scale,
    maxPixels: CONFIG.maxPixels
  });

  return ee.Number(result.get('DSWE1', 0));
}


function waterMedian(image, waterMask, region) {
  return image
    .select(commonBands)
    .updateMask(waterMask)
    .reduceRegion({
      reducer: ee.Reducer.median().unweighted(),
      geometry: region,
      scale: CONFIG.scale,
      maxPixels: CONFIG.maxPixels
    });
}


var output = samples.map(function(sample) {

  var geometry = sample.geometry();
  var coordinates = geometry.coordinates();
  var fieldTime = sampleDateTime(sample);
  var region = geometry.buffer(CONFIG.bufferRadiusMeters);

  var start = fieldTime.advance(
    -CONFIG.timeWindowHours,
    'hour'
  );

  var end = fieldTime.advance(
    CONFIG.timeWindowHours,
    'hour'
  );

  var candidates = landsat
    .filterBounds(region)
    .filterDate(start, end)
    .map(function(image) {

      var differenceMs = ee.Date(
        image.get('system:time_start')
      )
      .millis()
      .subtract(fieldTime.millis())
      .abs();

      return image.set({
        time_diff_ms: differenceMs,

        time_diff_hours:
          differenceMs.divide(1000 * 60 * 60),

        qa_valid_pixel_count:
          qaPixelCount(image, region)
      });
    })
    .filter(
      ee.Filter.gt(
        'qa_valid_pixel_count',
        0
      )
    );


  var base = sample.set({
    longitude: coordinates.get(0),
    latitude: coordinates.get(1),

    sample_datetime_local:
      fieldTime.format(
        'YYYY-MM-dd HH:mm:ss',
        CONFIG.timezone
      )
  });


  var noScene = base.set({
    match_status: 'no_match',
    match_reason: 'no_qa_valid_scene'
  });


  return ee.Feature(
    ee.Algorithms.If(
      candidates.size().eq(0),

      noScene,

      (function() {

        var image = ee.Image(
          candidates
            .sort('time_diff_ms')
            .first()
        );

        var landsatTime = ee.Date(
          image.get('system:time_start')
        );

        var waterMask = dswe1Mask(image);

        var waterCount = waterPixelCount(
          waterMask,
          region
        );

        var metadata = {
          landsat_datetime_utc:
            landsatTime.format(
              'YYYY-MM-dd HH:mm:ss',
              'UTC'
            ),

          time_diff_hours:
            image.get('time_diff_hours'),

          scene_id:
            image.get('LANDSAT_PRODUCT_ID'),

          sensor:
            image.get('sensor'),

          qa_valid_pixel_count:
            image.get('qa_valid_pixel_count'),

          dswe1_pixel_count:
            waterCount
        };


        var insufficientWater = base
          .set(metadata)
          .set({
            match_status: 'no_match',

            match_reason:
              'insufficient_dswe1_on_nearest_scene'
          });


        var matched = base
          .set(
            waterMedian(
              image,
              waterMask,
              region
            )
          )
          .set(metadata)
          .set({
            match_status: 'matched',

            match_reason:
              'nearest_scene_dswe1_pass'
          });


        return ee.Feature(
          ee.Algorithms.If(
            waterCount.gte(
              CONFIG.minDswe1Pixels
            ),
            matched,
            insufficientWater
          )
        );

      })()
    )
  );
});


var matched = output.filter(
  ee.Filter.eq(
    'match_status',
    'matched'
  )
);


print(
  'Raw field observations:',
  rawSamples.size()
);

print(
  'Valid-time observations:',
  samples.size()
);

print(
  'Matched observations:',
  matched.size()
);

print(
  'Insufficient DSWE1 pixels:',
  output
    .filter(
      ee.Filter.eq(
        'match_reason',
        'insufficient_dswe1_on_nearest_scene'
      )
    )
    .size()
);


[1, 2, 6, 24, 48].forEach(
  function(hours) {

    print(
      'Matched within <= ' + hours + ' h:',
      matched
        .filter(
          ee.Filter.lte(
            'time_diff_hours',
            hours
          )
        )
        .size()
    );
  }
);


var selectors = [
  CONFIG.idCol,
  CONFIG.programCol,
  CONFIG.dateCol,
  CONFIG.timeCol,
  CONFIG.obsCol,

  'longitude',
  'latitude',

  'sample_datetime_local',
  'landsat_datetime_utc',
  'time_diff_hours',

  'scene_id',
  'sensor',

  'qa_valid_pixel_count',
  'dswe1_pixel_count',

  'match_status',
  'match_reason',

  'Blue',
  'Green',
  'Red',
  'NIR',
  'SWIR1',
  'SWIR2'
];


Export.table.toDrive({
  collection: output,
  description: CONFIG.exportDescription,
  folder: CONFIG.exportFolder,
  fileNamePrefix: CONFIG.exportDescription,
  fileFormat: 'CSV',
  selectors: selectors
});
