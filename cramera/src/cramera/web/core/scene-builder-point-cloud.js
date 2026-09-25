/* Bounded scan references. Coordinates remain in the source frame and units. */
(function (global, factory) {
  'use strict';
  const exported = factory(global);
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
  else Object.assign(global, exported);
})(typeof window !== 'undefined' ? window : globalThis, function (global) {
  'use strict';

  // %% supported scalar records
  const LIMITS = Object.freeze({bytes:64 * 1024 * 1024, headerBytes:64 * 1024, headerLineBytes:8192, recordBytes:64 * 1024, fields:64, points:2000000, displayPoints:250000, components:1024});
  const FLOAT32_MAXIMUM = 3.4028234663852886e38;
  const DEFAULT_COLOR = 0xd6e3e8;
  const NUMBER_TOKEN = /^[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?|nan|inf(?:inity)?)$/i;
  const TYPES = Object.freeze({
    int8:{size:1, method:'getInt8', minimum:-128, maximum:127},
    uint8:{size:1, method:'getUint8', minimum:0, maximum:255},
    int16:{size:2, method:'getInt16', minimum:-32768, maximum:32767},
    uint16:{size:2, method:'getUint16', minimum:0, maximum:65535},
    int32:{size:4, method:'getInt32', minimum:-2147483648, maximum:2147483647},
    uint32:{size:4, method:'getUint32', minimum:0, maximum:4294967295},
    float32:{size:4, method:'getFloat32'}, float64:{size:8, method:'getFloat64'},
  });
  const PLY_TYPES = Object.freeze({char:'int8', uchar:'uint8', short:'int16', ushort:'uint16', int:'int32', uint:'uint32', float:'float32', double:'float64'});

  /** A scan does not satisfy the supported scalar point-record contract. */
  class PointCloudError extends Error {}

  function integer(token, description, maximum) {
    if (!/^\d{1,10}$/.test(token || '')) throw new PointCloudError(description + ' must be a nonnegative integer.');
    const value = Number(token);
    if (!Number.isSafeInteger(value) || value > maximum) throw new PointCloudError(description + ' exceeds the supported limit.');
    return value;
  }

  function header(buffer, format) {
    const bytes = new Uint8Array(buffer);
    const decoder = new TextDecoder('utf-8', {fatal:true});
    const lines = [];
    let start = 0;
    const maximum = Math.min(bytes.length, LIMITS.headerBytes);
    for (let cursor = 0; cursor < maximum; cursor += 1) {
      if (bytes[cursor] !== 10) continue;
      if (cursor - start > LIMITS.headerLineBytes || bytes.subarray(start, cursor).some(value => value === 0 || value > 127)) throw new PointCloudError('Point-cloud headers must use bounded ASCII lines.');
      const line = decoder.decode(bytes.subarray(start, cursor)).trim();
      lines.push(line); start = cursor + 1;
      if ((format === 'ply' && line === 'end_header') || (format === 'pcd' && /^DATA\s+/i.test(line))) return {lines, length:start};
    }
    throw new PointCloudError('The point-cloud header is incomplete or too large.');
  }

  /** Iterate bounded source records without allocating a full cloud geometry. */
  class PointCloudRecords {
    constructor(buffer, format) {
      if (!(buffer instanceof ArrayBuffer) || buffer.byteLength === 0 || buffer.byteLength > LIMITS.bytes) throw new PointCloudError('Point clouds must be nonempty and at most 64 MiB.');
      if (!['ply','pcd'].includes(format)) throw new PointCloudError('Choose a PLY or PCD point cloud.');
      this.buffer = buffer;
      this.format = format;
      this.fields = new Map();
      this.components = 0;
      this.stride = 0;
      const sourceHeader = header(buffer, format);
      this.bodyOffset = sourceHeader.length;
      if (format === 'ply') this.readPlyHeader(sourceHeader.lines);
      else this.readPcdHeader(sourceHeader.lines);
      if (!this.pointCount) throw new PointCloudError('The point cloud contains no points.');
      for (const name of ['x','y','z']) {
        if (!this.fields.has(name) || this.fields.get(name).count !== 1) throw new PointCloudError('Point clouds require scalar x, y and z fields.');
      }
      if (this.encoding !== 'ascii' && buffer.byteLength - this.bodyOffset !== this.pointCount * this.stride) throw new PointCloudError('Binary point data does not match the declared point count.');
      this.view = new DataView(buffer);
      this.packedColor = this.fields.get('rgb') || this.fields.get('rgba');
      if (this.packedColor && (this.packedColor.count !== 1 || this.packedColor.type.size !== 4)) throw new PointCloudError('Packed RGB colors require one 32-bit scalar.');
      this.colorNames = this.fields.has('red') && this.fields.has('green') && this.fields.has('blue') ? ['red','green','blue'] : ['r','g','b'];
      this.hasColor = Boolean(this.packedColor || this.colorNames.every(name => this.fields.has(name) && this.fields.get(name).count === 1));
      this.colorBits = new DataView(new ArrayBuffer(4));
    }

    addField(name, typeName, count = 1) {
      const type = Object.prototype.hasOwnProperty.call(TYPES, typeName) ? TYPES[typeName] : null;
      if (!type || !/^[A-Za-z_][A-Za-z0-9_]{0,127}$/.test(name) || this.fields.has(name) || count < 1) throw new PointCloudError('The point-cloud field declaration is invalid.');
      if (this.fields.size >= LIMITS.fields) throw new PointCloudError('The point cloud has too many fields.');
      if (this.components + count > LIMITS.components) throw new PointCloudError('A point has too many scalar components.');
      this.fields.set(name, {name, type, typeName, count, component:this.components, offset:this.stride});
      this.components += count; this.stride += type.size * count;
    }

    readPlyHeader(lines) {
      if (lines[0] !== 'ply') throw new PointCloudError('The file is not a PLY point cloud.');
      let element = null;
      let foundVertices = false;
      for (const line of lines.slice(1)) {
        const tokens = line.split(/\s+/);
        const command = tokens[0];
        if (!line) throw new PointCloudError('Empty PLY header declarations are unsupported.');
        if (['comment','obj_info','end_header'].includes(command)) continue;
        if (command === 'format') {
          if (this.encoding || tokens[2] !== '1.0' || tokens.length !== 3) throw new PointCloudError('The PLY format declaration is invalid.');
          this.encoding = tokens[1];
          if (!['ascii','binary_little_endian','binary_big_endian'].includes(this.encoding)) throw new PointCloudError('Unsupported PLY encoding.');
          this.littleEndian = this.encoding === 'binary_little_endian';
        } else if (command === 'element') {
          if (tokens.length !== 3 || !this.encoding) throw new PointCloudError('The PLY element declaration is invalid.');
          element = tokens[1];
          const count = integer(tokens[2], 'Point count', LIMITS.points);
          if (element !== 'vertex' && count !== 0) throw new PointCloudError('Import a vertex-only PLY point cloud; mesh faces are not supported.');
          if (element === 'vertex') {
            if (foundVertices) throw new PointCloudError('The PLY vertex element is duplicated.');
            foundVertices = true; this.pointCount = count;
          }
        } else if (command === 'property') {
          if (!element) throw new PointCloudError('PLY properties require an element.');
          if (element !== 'vertex') continue;
          if (tokens.length !== 3 || tokens[1] === 'list') throw new PointCloudError('Point-cloud vertices must have scalar properties.');
          this.addField(tokens[2], PLY_TYPES[tokens[1]] || tokens[1]);
        } else throw new PointCloudError('Unsupported PLY header entry: ' + command);
      }
      if (!this.encoding || !foundVertices) throw new PointCloudError('The PLY format or vertex count is missing.');
    }

    readPcdHeader(lines) {
      const declarations = new Map();
      const allowed = ['VERSION','FIELDS','SIZE','TYPE','COUNT','WIDTH','HEIGHT','VIEWPOINT','POINTS','DATA'];
      for (const source of lines) {
        const line = source.split('#', 1)[0].trim();
        if (!line) continue;
        const tokens = line.split(/\s+/);
        const key = tokens.shift();
        if (!allowed.includes(key) || declarations.has(key)) throw new PointCloudError('The PCD header contains unknown or duplicate entries.');
        declarations.set(key, tokens);
      }
      for (const required of ['VERSION','FIELDS','SIZE','TYPE','WIDTH','HEIGHT','POINTS','DATA']) {
        if (!declarations.has(required)) throw new PointCloudError('The PCD header is incomplete.');
      }
      for (const scalar of ['VERSION','WIDTH','HEIGHT','POINTS','DATA']) {
        if (declarations.get(scalar).length !== 1) throw new PointCloudError('A PCD scalar declaration has extra values.');
      }
      if (!['.7','0.7'].includes(declarations.get('VERSION')[0])) throw new PointCloudError('Only PCD version 0.7 is supported.');
      if (declarations.has('VIEWPOINT')) {
        const viewpoint = declarations.get('VIEWPOINT');
        if (viewpoint.length !== 7 || viewpoint.some(token => token.length > 128 || !NUMBER_TOKEN.test(token) || !Number.isFinite(Number(token)))) throw new PointCloudError('PCD VIEWPOINT requires seven finite numbers.');
      }
      const fields = declarations.get('FIELDS');
      const sizes = declarations.get('SIZE');
      const types = declarations.get('TYPE');
      if (!fields || !sizes || !types || fields.length !== sizes.length || fields.length !== types.length) throw new PointCloudError('PCD FIELDS, SIZE and TYPE declarations must agree.');
      const counts = declarations.get('COUNT') || fields.map(() => '1');
      if (counts.length !== fields.length) throw new PointCloudError('PCD COUNT must describe every field.');
      fields.forEach((name, index) => {
        const size = integer(sizes[index], 'Scalar size', 8);
        const prefix = {I:'int', U:'uint', F:'float'}[types[index]];
        this.addField(name, prefix + size * 8, integer(counts[index], 'Field count', LIMITS.components));
      });
      const width = declarations.has('WIDTH') ? integer(declarations.get('WIDTH')[0], 'Cloud width', LIMITS.points) : null;
      const height = declarations.has('HEIGHT') ? integer(declarations.get('HEIGHT')[0], 'Cloud height', LIMITS.points) : null;
      const count = declarations.get('POINTS');
      this.pointCount = count ? integer(count[0], 'Point count', LIMITS.points) : width * height;
      if (!Number.isSafeInteger(this.pointCount) || this.pointCount > LIMITS.points || (width !== null && height !== null && width * height !== this.pointCount)) throw new PointCloudError('PCD dimensions and point count do not agree.');
      const data = declarations.get('DATA');
      this.encoding = data && data[0];
      if (!['ascii','binary'].includes(this.encoding)) throw new PointCloudError('Use ASCII or uncompressed binary PCD data.');
      this.littleEndian = true;
    }

    number(token, field) {
      if (token.length > 128 || !NUMBER_TOKEN.test(token)) throw new PointCloudError('A point-cloud value is not a bounded numeric token.');
      const lower = token.toLowerCase();
      const value = /nan/.test(lower) ? NaN : /inf/.test(lower) ? (lower.startsWith('-') ? -Infinity : Infinity) : Number(token);
      if (field.type.minimum !== undefined && (!/^[+-]?\d+$/.test(token) || !Number.isInteger(value) || value < field.type.minimum || value > field.type.maximum)) throw new PointCloudError('An integer value exceeds its declared scalar type.');
      if (field.typeName === 'float32' && Number.isFinite(value) && Math.abs(value) > FLOAT32_MAXIMUM) throw new PointCloudError('A float value exceeds its declared scalar type.');
      return value;
    }

    forEach(visit) {
      if (this.encoding !== 'ascii') {
        for (let index = 0; index < this.pointCount; index += 1) {
          this.recordOffset = this.bodyOffset + index * this.stride;
          visit(this);
        }
        return;
      }
      const text = new TextDecoder('utf-8', {fatal:true}).decode(new Uint8Array(this.buffer, this.bodyOffset));
      let start = 0;
      let count = 0;
      while (start < text.length) {
        let end = text.indexOf('\n', start);
        if (end < 0) end = text.length;
        if (end - start + (end < text.length ? 1 : 0) > LIMITS.recordBytes) throw new PointCloudError('The point-cloud record is too long.');
        const line = text.slice(start, end).trim(); start = end + 1;
        if (/[^\x00-\x7f]/.test(line)) throw new PointCloudError('ASCII point records contain non-ASCII characters.');
        if (!line) continue;
        const tokens = line.split(/\s+/);
        if (count >= this.pointCount || tokens.length !== this.components) throw new PointCloudError('ASCII point data does not match the declared record shape.');
        this.values = new Array(tokens.length);
        this.fields.forEach(field => {
          for (let component = 0; component < field.count; component += 1) {
            const offset = field.component + component;
            this.values[offset] = this.number(tokens[offset], field);
          }
        });
        visit(this); count += 1;
      }
      if (count !== this.pointCount) throw new PointCloudError('ASCII point data is truncated.');
    }

    value(name) {
      const field = this.fields.get(name);
      return this.encoding === 'ascii' ? this.values[field.component] : this.view[field.type.method](this.recordOffset + field.offset, this.littleEndian);
    }

    position() {
      const coordinates = ['x','y','z'].map(name => this.value(name));
      if (coordinates.some(value => !Number.isNaN(value) && (!Number.isFinite(value) || Math.abs(value) > FLOAT32_MAXIMUM))) throw new PointCloudError('A coordinate exceeds the finite Float32 range.');
      return coordinates.every(Number.isFinite) ? coordinates : null;
    }

    color() {
      if (!this.hasColor) return null;
      if (this.packedColor) {
        let packed;
        if (this.encoding !== 'ascii') packed = this.view.getUint32(this.recordOffset + this.packedColor.offset, this.littleEndian);
        else if (this.packedColor.typeName === 'float32') { this.colorBits.setFloat32(0, this.value(this.packedColor.name), true); packed = this.colorBits.getUint32(0, true); }
        else packed = this.value(this.packedColor.name) >>> 0;
        return [(packed >>> 16 & 255) / 255, (packed >>> 8 & 255) / 255, (packed & 255) / 255];
      }
      return this.colorNames.map(name => {
        const value = this.value(name);
        const maximum = this.fields.get(name).typeName.startsWith('float') ? 1 : 255;
        return Number.isFinite(value) ? Math.min(1, Math.max(0, value / maximum)) : 1;
      });
    }
  }

  // %% shared preflight and display sampling
  class PointCloudSupport {
    static LIMITS = LIMITS;
    static DEFAULT_COLOR = DEFAULT_COLOR;

    /** Validate every source record before uploading, without creating geometry. */
    static inspect(buffer, format) {
      const records = new PointCloudRecords(buffer, format);
      let finitePointCount = 0;
      records.forEach(record => { if (record.position()) finitePointCount += 1; });
      if (finitePointCount === 0) throw new PointCloudError('The point cloud contains no finite XYZ measurements.');
      const stride = Math.ceil(finitePointCount / LIMITS.displayPoints);
      return {format, pointCount:records.pointCount, finitePointCount, displayPointCount:Math.ceil(finitePointCount / stride), hasColor:records.hasColor};
    }

    /** Draw a deterministic finite subset while retaining the original coordinates. */
    static parse(buffer, format, THREE = global.THREE) {
      const metadata = this.inspect(buffer, format);
      const records = new PointCloudRecords(buffer, format);
      const positions = new Float32Array(metadata.displayPointCount * 3);
      const colors = metadata.hasColor ? new Float32Array(positions.length) : null;
      const color = new THREE.Color();
      const stride = Math.ceil(metadata.finitePointCount / LIMITS.displayPoints);
      let finiteIndex = 0;
      let displayed = 0;
      records.forEach(record => {
        const position = record.position();
        if (!position) return;
        if (finiteIndex % stride === 0) {
          positions.set(position, displayed * 3);
          if (colors) {
            color.setRGB(...record.color()).convertSRGBToLinear();
            colors.set([color.r, color.g, color.b], displayed * 3);
          }
          displayed += 1;
        }
        finiteIndex += 1;
      });
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      if (colors) geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      geometry.computeBoundingSphere(); geometry.computeBoundingBox();
      const material = new THREE.PointsMaterial({size:.006, sizeAttenuation:true, vertexColors:Boolean(colors), color:colors ? 0xffffff : DEFAULT_COLOR, transparent:true, opacity:.65, depthWrite:false, toneMapped:false});
      const points = new THREE.Points(geometry, material);
      points.userData.pointCloud = metadata;
      return points;
    }
  }

  return {PointCloudSupport, PointCloudError};
});
