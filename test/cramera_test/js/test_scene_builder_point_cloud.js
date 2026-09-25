'use strict';

// %% shared scan fixtures
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const THREE = require('../../../cramera/src/cramera/web/vendor/three.min.js');
const {PointCloudSupport, PointCloudError} = require('../../../cramera/src/cramera/web/core/scene-builder-point-cloud.js');
const FIXTURES = path.join(__dirname, '../dataset/point_clouds');

function fixture(name) {
  const buffer = fs.readFileSync(path.join(FIXTURES, name));
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}

for (const name of ['colored_ascii.ply', 'colored_binary_le.ply', 'colored_binary_be.ply', 'colored_ascii.pcd', 'colored_binary.pcd']) {
  test(name + ' retains world coordinates and RGB colors', () => {
    const format = name.split('.').pop();
    const cloud = PointCloudSupport.parse(fixture(name), format, THREE);
    assert.deepEqual(Array.from(cloud.geometry.attributes.position.array), [1,2,3,-4,.5,6,7,8,9]);
    assert.deepEqual(Array.from(cloud.geometry.attributes.color.array), [1,0,0,0,1,0,0,0,1]);
    assert.equal(cloud.material.depthWrite, false);
    assert.equal(cloud.userData.pointCloud.pointCount, 3);
    assert.equal(cloud.userData.pointCloud.finitePointCount, 3);
    cloud.geometry.dispose(); cloud.material.dispose();
  });
}

test('missing depth measurements are filtered without moving the valid sample', () => {
  const cloud = PointCloudSupport.parse(fixture('missing_measurements.pcd'), 'pcd', THREE);
  assert.deepEqual(Array.from(cloud.geometry.attributes.position.array), [4,5,6]);
  assert.equal(cloud.userData.pointCloud.pointCount, 3);
  assert.equal(cloud.userData.pointCloud.finitePointCount, 1);
  assert.equal(cloud.geometry.attributes.color, undefined);
  assert.equal(cloud.material.color.getHex(), PointCloudSupport.DEFAULT_COLOR);
});

test('preflight validates the complete source without allocating THREE objects', () => {
  const result = PointCloudSupport.inspect(fixture('colored_binary_le.ply'), 'ply');
  assert.deepEqual(result, {format:'ply',pointCount:3,finitePointCount:3,displayPointCount:3,hasColor:true});
});

test('midgray scan colors retain sRGB appearance in the lit editor viewport', () => {
  const source = altered('colored_ascii.ply', text => text.replace('255 0 0', '128 128 128'));
  const cloud = PointCloudSupport.parse(source, 'ply', THREE);
  const expected = new THREE.Color(128 / 255, 128 / 255, 128 / 255).convertSRGBToLinear();
  assert.equal(cloud.geometry.attributes.color.getX(0), Math.fround(expected.r));
  assert.equal(cloud.material.toneMapped, false);
});

test('truncated binary point data is rejected before rendering', () => {
  const data = fixture('colored_binary.pcd');
  assert.throws(() => PointCloudSupport.parse(data.slice(0, -1), 'pcd', THREE), PointCloudError);
});

test('mismatched source formats and unsupported formats are rejected', () => {
  assert.throws(() => PointCloudSupport.inspect(fixture('colored_ascii.ply'), 'pcd'), PointCloudError);
  assert.throws(() => PointCloudSupport.inspect(fixture('colored_ascii.ply'), 'obj'), PointCloudError);
});

// %% preflight boundaries and structured headers
function altered(name, transform) {
  return new TextEncoder().encode(transform(new TextDecoder().decode(fixture(name)))).buffer;
}

for (const [name, transform] of [
  ['compressed PCD', text => text.replace('DATA ascii', 'DATA binary_compressed')],
  ['unknown PCD version', text => text.replace('VERSION 0.7', 'VERSION 0.8')],
  ['missing PCD version', text => text.replace('VERSION 0.7\n', '')],
  ['extra scalar declaration values', text => text.replace('WIDTH 3', 'WIDTH 3 4')],
  ['unknown header directives', text => text.replace('VERSION 0.7', 'UNSAFE xyz\nVERSION 0.7')],
  ['invalid viewpoint', text => text.replace('VIEWPOINT 0 0 0 1 0 0 0', 'VIEWPOINT 0 0 0 nan 0 0 0')],
  ['infinite coordinates', text => text.replace('1.0 2.0 3.0', 'inf 2.0 3.0')],
  ['Float32 coordinate overflow', text => text.replace('1.0 2.0 3.0', '1e39 2.0 3.0')],
  ['excessive declared points', text => text.replace('POINTS 3', 'POINTS 2000001').replace('WIDTH 3', 'WIDTH 2000001')],
]) {
  test(name + ' is rejected before upload', () => {
    assert.throws(() => PointCloudSupport.inspect(altered('colored_ascii.pcd', transform), 'pcd'), PointCloudError);
  });
}

test('nonempty PLY faces and vertex lists are explicitly rejected', () => {
  assert.throws(() => PointCloudSupport.inspect(altered('colored_ascii.ply', text => text.replace('end_header', 'element face 1\nproperty list uchar int vertex_indices\nend_header')), 'ply'), PointCloudError);
  assert.throws(() => PointCloudSupport.inspect(altered('colored_ascii.ply', text => text.replace('property float x', 'property list uchar float x')), 'ply'), PointCloudError);
});

test('oversized uploads are rejected before reading their header', () => {
  assert.throws(() => PointCloudSupport.inspect(new ArrayBuffer(PointCloudSupport.LIMITS.bytes + 1), 'ply'), PointCloudError);
});

test('declared integer fields require integer tokens before upload', () => {
  assert.throws(() => PointCloudSupport.inspect(altered('colored_ascii.ply', text => text.replace('255 0 0', '255.0 0 0')), 'ply'), PointCloudError);
});

test('numeric tokens have the same bounded length as the server parser', () => {
  assert.throws(() => PointCloudSupport.inspect(altered('colored_ascii.pcd', text => text.replace('1.0 2.0 3.0', '0'.repeat(129) + ' 2.0 3.0')), 'pcd'), PointCloudError);
});

test('PCD COUNT offsets preserve coordinates after vector-valued features', () => {
  const source = altered('colored_ascii.pcd', text => text.replace('FIELDS x y z rgb', 'FIELDS descriptor x y z rgb').replace('SIZE 4 4 4 4', 'SIZE 4 4 4 4 4').replace('TYPE F F F F', 'TYPE F F F F F').replace('COUNT 1 1 1 1', 'COUNT 2 1 1 1 1').replace(/^(1\.0 2\.0 3\.0|-4\.0 0\.5 6\.0|7\.0 8\.0 9\.0)/gm, '99 100 $1'));
  const cloud = PointCloudSupport.parse(source, 'pcd', THREE);
  assert.deepEqual(Array.from(cloud.geometry.attributes.position.array), [1,2,3,-4,.5,6,7,8,9]);
  assert.deepEqual(Array.from(cloud.geometry.attributes.color.array), [1,0,0,0,1,0,0,0,1]);
});

test('large finite clouds are deterministically downsampled without recentering', () => {
  const pointCount = PointCloudSupport.LIMITS.displayPoints + 1;
  const original = new Uint8Array(fixture('colored_binary_le.ply'));
  const text = new TextDecoder().decode(original);
  const end = text.indexOf('end_header\n') + 'end_header\n'.length;
  const header = new TextEncoder().encode(text.slice(0, end).replace('vertex 3', 'vertex ' + pointCount).replace(/property uchar (red|green|blue)\n/g, ''));
  const data = new Uint8Array(header.length + pointCount * 12); data.set(header);
  const values = new DataView(data.buffer);
  for (let index = 0; index < pointCount; index += 1) values.setFloat32(header.length + index * 12, index + 1000, true);
  const first = PointCloudSupport.parse(data.buffer, 'ply', THREE);
  const second = PointCloudSupport.parse(data.buffer, 'ply', THREE);
  const stride = Math.ceil(pointCount / PointCloudSupport.LIMITS.displayPoints);
  const positions = first.geometry.attributes.position;
  assert.equal(positions.count, Math.ceil(pointCount / stride));
  assert.equal(positions.getX(0), 1000);
  assert.equal(positions.getX(1), 1000 + stride);
  assert.deepEqual(first.geometry.attributes.position.array, second.geometry.attributes.position.array);
});
